#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.optimizers.fire2 import FIRE2
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.models.aimnet2 import AIMNet2Wrapper

OUT=Path("03-validation"); OUT.mkdir(exist_ok=True)
DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"

SITE_SMILES="OCC[N+](C)(C)Cc1ccccc1"
IONS={"carbonate_control":"[O-]C(=O)[O-]","hpo4_candidate":"OP(=O)([O-])[O-]"}
HYDRATION={"dry_0h2o":0,"wet_6h2o":6}
SITE_N_TARGETS=[np.array([-4.5,0.0,0.0]),np.array([4.5,0.0,0.0])]
MIN_INTERFRAG_A=1.45
N_TRIALS=2
CORE_FMAX=0.08
FINAL_FMAX=0.08
ACCEPT_FMAX=0.30
CORE_STEPS=800
FINAL_STEPS=700
DT=0.001
NEAR_SHELL=(4.2,5.5)
FAR_SHELL=(10.0,12.0)

def embed(smiles,seed):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    if m is None: raise ValueError(smiles)
    p=AllChem.ETKDGv3(); p.randomSeed=int(seed)
    if AllChem.EmbedMolecule(m,p)!=0: raise RuntimeError(f"embed failed: {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(m): AllChem.MMFFOptimizeMolecule(m,maxIters=800)
    c=m.GetConformer()
    pos=np.array([list(c.GetAtomPosition(i)) for i in range(m.GetNumAtoms())],dtype=np.float32)
    z=np.array([a.GetAtomicNum() for a in m.GetAtoms()],dtype=np.int64)
    q=int(sum(a.GetFormalCharge() for a in m.GetAtoms()))
    return pos,z,q

def place_atom(pos, atom_idx, target):
    return pos-pos[atom_idx]+np.asarray(target,dtype=np.float32)

def min_cross(a,b):
    d=a[:,None,:]-b[None,:,:]
    return float(np.sqrt((d*d).sum(axis=2)).min())

def place_safe(fragment,existing,rng,shell,min_dist=1.55,attempts=1000):
    for _ in range(attempts):
        v=rng.normal(size=3); v/=np.linalg.norm(v)
        r=rng.uniform(shell[0],shell[1])
        cand=fragment-fragment.mean(axis=0,keepdims=True)+v*r
        if all(min_cross(cand,e)>=min_dist for e in existing): return cand
    raise RuntimeError(f"safe placement failed shell={shell}")

def make_batch(systems):
    return Batch.from_data_list(systems,device=DEVICE)

def relax(batch,model,steps,fmax):
    opt=FIRE2(model=model,dt=DT,n_steps=steps,
              convergence_hook=ConvergenceHook.from_fmax(threshold=fmax,source_status=0,target_status=1))
    for hook in model.make_neighbor_hooks():
        opt.register_hook(hook,stage=DynamicsStage.BEFORE_COMPUTE)
    return opt.run(batch)

def graph_slices(batch,meta):
    pos=batch.positions.detach().cpu().numpy()
    z=batch.atomic_numbers.detach().cpu().numpy()
    out=[]; cur=0
    for m in meta:
        n=m["n_atoms"]
        out.append((pos[cur:cur+n].copy(),z[cur:cur+n].copy()))
        cur+=n
    return out

def fmax_per_graph(batch):
    forces=batch.forces.detach().cpu()
    bidx=batch.batch_idx.detach().cpu()
    fn=forces.norm(dim=-1)
    fmax=torch.zeros(batch.num_graphs)
    fmax.scatter_reduce_(0,bidx,fn,reduce="amax",include_self=True)
    return fmax.numpy()

site0,site_z,site_q=embed(SITE_SMILES,101)
nidx=np.where(site_z==7)[0]
if site_q!=1 or len(nidx)!=1: raise SystemExit("site sanity failed")
nidx=int(nidx[0])
sites=[place_atom(site0,nidx,t) for t in SITE_N_TARGETS]
if min_cross(sites[0],sites[1])<MIN_INTERFRAG_A: raise SystemExit("site-site overlap")
co2_0,co2_z,co2_q=embed("O=C=O",202)
if co2_q!=0: raise SystemExit("CO2 charge sanity failed")

# Stage 1: rebuild exactly the D1-safe cores and relax them first.
cores=[]; core_meta=[]
for ci,(name,smi) in enumerate(IONS.items()):
    ion0,ion_z,ion_q=embed(smi,500+ci)
    if 2*site_q+ion_q!=0: raise SystemExit(f"charge mismatch {name}")
    for hname,nw in HYDRATION.items():
        for trial in range(N_TRIALS):
            rng=np.random.default_rng(310000+ci*1000+nw*100+trial)
            existing=[sites[0],sites[1]]
            ion=place_safe(ion0,existing,rng,(3.0,5.0),1.55)
            existing.append(ion)
            parts=[sites[0],sites[1],ion]; nums=[site_z,site_z,ion_z]
            for wi in range(nw):
                w0,wz,_=embed("O",1000+ci*100+trial*10+wi)
                w=place_safe(w0,existing,rng,(5.5,8.5),1.55)
                existing.append(w); parts.append(w); nums.append(wz)
            pos=np.concatenate(parts); z=np.concatenate(nums)
            cores.append(AtomicData(
                positions=torch.tensor(pos,dtype=torch.float32),
                atomic_numbers=torch.tensor(z,dtype=torch.long),
                forces=torch.zeros(len(z),3),energy=torch.zeros(1,1),
                charge=torch.zeros(1,1),velocities=torch.zeros(len(z),3)))
            core_meta.append({"candidate":name,"hydration":hname,"trial":trial,"n_atoms":len(z)})

model=AIMNet2Wrapper.from_checkpoint("aimnet2_wb97m_d3_3",device=DEVICE,compile_model=False).eval()
model.model_config.active_outputs={"energy","forces"}
core_batch=relax(make_batch(cores),model,CORE_STEPS,CORE_FMAX)
core_fmax=fmax_per_graph(core_batch)
relaxed_cores=graph_slices(core_batch,core_meta)

# Hard gate: all D1-equivalent cores must still be acceptable before CO2 is added.
core_accept=[bool(np.isfinite(x) and x<=ACCEPT_FMAX) for x in core_fmax]
if not all(core_accept):
    report={"status":"FAIL_CORE_RECONVERGENCE","core_fmax":[float(x) for x in core_fmax],"core_accept":core_accept}
    (OUT/"alchemi-ira910-gate-d2-co2.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); raise SystemExit(2)

# Stage 2: add CO2 only to already-relaxed cores, with collision-free near/far placement.
systems=[]; meta=[]
for i,((core_pos,core_z),cm) in enumerate(zip(relaxed_cores,core_meta)):
    rng=np.random.default_rng(910000+i)
    for state,shell in [("near",NEAR_SHELL),("far",FAR_SHELL)]:
        cp=place_safe(co2_0,[core_pos],rng,shell,1.55)
        pos=np.concatenate([core_pos,cp]); z=np.concatenate([core_z,co2_z])
        systems.append(AtomicData(
            positions=torch.tensor(pos,dtype=torch.float32),
            atomic_numbers=torch.tensor(z,dtype=torch.long),
            forces=torch.zeros(len(z),3),energy=torch.zeros(1,1),
            charge=torch.zeros(1,1),velocities=torch.zeros(len(z),3)))
        meta.append({**cm,"start_state":state,"n_atoms":len(z)})

batch=relax(make_batch(systems),model,FINAL_STEPS,FINAL_FMAX)
fmax=fmax_per_graph(batch)
energies=batch.energy.reshape(-1).detach().cpu().numpy()

raw=[]
for i,m in enumerate(meta):
    fm=float(fmax[i])
    raw.append({**m,"final_fmax_eV_A":fm,"final_energy_eV":float(energies[i]),
                "accepted":bool(np.isfinite(fm) and fm<=ACCEPT_FMAX)})

summary={}
for cand in IONS:
    summary[cand]={}
    for h in HYDRATION:
        vals=[]; details=[]
        for t in range(N_TRIALS):
            n=next(x for x in raw if x["candidate"]==cand and x["hydration"]==h and x["trial"]==t and x["start_state"]=="near")
            f=next(x for x in raw if x["candidate"]==cand and x["hydration"]==h and x["trial"]==t and x["start_state"]=="far")
            ok=n["accepted"] and f["accepted"]
            delta=n["final_energy_eV"]-f["final_energy_eV"]
            details.append({"trial":t,"accepted_pair":ok,"delta_eV":float(delta),
                            "near_fmax":n["final_fmax_eV_A"],"far_fmax":f["final_fmax_eV_A"]})
            if ok: vals.append(delta)
        a=np.asarray(vals,float)
        summary[cand][h]={
            "valid_pairs":len(vals),"pair_details":details,
            "median_delta_eV":float(np.median(a)) if len(a) else None,
            "mean_delta_eV":float(a.mean()) if len(a) else None}
    d=summary[cand]["dry_0h2o"]["median_delta_eV"]
    w=summary[cand]["wet_6h2o"]["median_delta_eV"]
    summary[cand]["hydration_shift_eV"]=float(w-d) if d is not None and w is not None else None

hpo=summary["hpo4_candidate"]; carb=summary["carbonate_control"]
usable=all(summary[c][h]["valid_pairs"]>=1 for c in IONS for h in HYDRATION)
report={
 "status":"PASS" if usable else "INCONCLUSIVE_CONVERGENCE",
 "device":DEVICE,"cuda_available":torch.cuda.is_available(),
 "test":"Gate D.2 IRA910 Type-II HPO4 vs carbonate after D1-safe core relaxation",
 "core_systems":len(cores),"core_accepted":sum(core_accept),
 "co2_systems":len(systems),"accepted_systems":sum(x["accepted"] for x in raw),
 "summary":summary,"raw":raw,
 "comparison":{
   "hpo4_dry_median_delta_eV":hpo["dry_0h2o"]["median_delta_eV"],
   "hpo4_wet_median_delta_eV":hpo["wet_6h2o"]["median_delta_eV"],
   "hpo4_hydration_shift_eV":hpo["hydration_shift_eV"],
   "carbonate_dry_median_delta_eV":carb["dry_0h2o"]["median_delta_eV"],
   "carbonate_wet_median_delta_eV":carb["wet_6h2o"]["median_delta_eV"],
   "carbonate_hydration_shift_eV":carb["hydration_shift_eV"]},
 "guardrails":[
   "Local two-site Type-II motif only; not full IRA910 polymer/pore.",
   "Near-minus-far relaxed energy is a screening proxy, not adsorption free energy.",
   "Microhydration is not relative humidity.",
   "Two trials per state are a focused Gate D2 check, not statistical validation.",
   "No discovery or lab validation claim is allowed from this result alone.",
   "Independent computational cross-check is required before lab promotion."],
 "next_gate":"If convergence is usable, interpret only sign/robustness versus carbonate and then cross-check independently before scaling atomistic calculations."
}
(OUT/"alchemi-ira910-gate-d2-co2.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))

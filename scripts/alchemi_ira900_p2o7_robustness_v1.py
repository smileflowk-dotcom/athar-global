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

SITE_SMILES="C[N+](C)(C)Cc1ccccc1"  # IRA900 Type-I local site
IONS={
  "carbonate":{"smiles":"[O-]C(=O)[O-]","charge":-2},
  "p2o7":{"smiles":"[O-]P(=O)([O-])OP(=O)([O-])[O-]","charge":-4},
}
HYDRATION={"dry_0h2o":0,"low_3h2o":3,"mid_6h2o":6,"high_9h2o":9}
N_TRIALS=3
CORE_FMAX=0.08
FINAL_FMAX=0.08
ACCEPT_FMAX=0.30
CORE_STEPS=900
FINAL_STEPS=700
DT=0.001
NEAR_SHELL=(4.2,5.5)
FAR_SHELL=(10.0,12.0)
MIN_DIST=1.55

def embed(smiles,seed):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    if m is None: raise ValueError(smiles)
    p=AllChem.ETKDGv3(); p.randomSeed=int(seed)
    if AllChem.EmbedMolecule(m,p)!=0: raise RuntimeError(f"embed failed: {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(m): AllChem.MMFFOptimizeMolecule(m,maxIters=1000)
    c=m.GetConformer()
    pos=np.array([list(c.GetAtomPosition(i)) for i in range(m.GetNumAtoms())],dtype=np.float32)
    z=np.array([a.GetAtomicNum() for a in m.GetAtoms()],dtype=np.int64)
    q=int(sum(a.GetFormalCharge() for a in m.GetAtoms()))
    return pos,z,q

def place_atom(pos,idx,target):
    return pos-pos[idx]+np.asarray(target,dtype=np.float32)

def min_cross(a,b):
    d=a[:,None,:]-b[None,:,:]
    return float(np.sqrt((d*d).sum(axis=2)).min())

def safe_shell(fragment,existing,rng,shell,min_dist=MIN_DIST,attempts=2500):
    for _ in range(attempts):
        v=rng.normal(size=3); v/=np.linalg.norm(v)
        r=rng.uniform(shell[0],shell[1])
        cand=fragment-fragment.mean(axis=0,keepdims=True)+v*r
        if all(min_cross(cand,e)>=min_dist for e in existing): return cand
    raise RuntimeError(f"safe placement failed shell={shell}")

def site_vectors(n,r=6.2):
    if n==2:
        u=np.array([[1,0,0],[-1,0,0]],float)
    elif n==4:
        u=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)
        u=u/np.linalg.norm(u,axis=1,keepdims=True)
    else: raise ValueError(n)
    return u*r

def make_batch(xs): return Batch.from_data_list(xs,device=DEVICE)

def relax(batch,model,steps,fmax):
    opt=FIRE2(model=model,dt=DT,n_steps=steps,
      convergence_hook=ConvergenceHook.from_fmax(threshold=fmax,source_status=0,target_status=1))
    for hook in model.make_neighbor_hooks():
        opt.register_hook(hook,stage=DynamicsStage.BEFORE_COMPUTE)
    return opt.run(batch)

def fmax_per_graph(batch):
    forces=batch.forces.detach().cpu(); bidx=batch.batch_idx.detach().cpu()
    fn=forces.norm(dim=-1); fmax=torch.zeros(batch.num_graphs)
    fmax.scatter_reduce_(0,bidx,fn,reduce="amax",include_self=True)
    return fmax.numpy()

def graph_slices(batch,meta):
    p=batch.positions.detach().cpu().numpy(); z=batch.atomic_numbers.detach().cpu().numpy()
    out=[]; cur=0
    for m in meta:
        n=m["n_atoms"]; out.append((p[cur:cur+n].copy(),z[cur:cur+n].copy())); cur+=n
    return out

site0,site_z,site_q=embed(SITE_SMILES,101)
nidx=np.where(site_z==7)[0]
if site_q!=1 or len(nidx)!=1: raise SystemExit("IRA900 Type-I site sanity failed")
nidx=int(nidx[0])
co2_0,co2_z,co2_q=embed("O=C=O",202)
if co2_q!=0: raise SystemExit("CO2 sanity failed")

cores=[]; core_meta=[]
for ci,(name,spec) in enumerate(IONS.items()):
    ion0,ion_z,ion_q=embed(spec["smiles"],500+ci)
    if ion_q!=spec["charge"]: raise SystemExit(f"ion charge mismatch {name}: {ion_q}")
    nsites=abs(ion_q); targets=site_vectors(nsites)
    sites=[place_atom(site0,nidx,t) for t in targets]
    for hname,nw in HYDRATION.items():
        for trial in range(N_TRIALS):
            rng=np.random.default_rng(1200000+ci*100000+nw*1000+trial)
            ion=safe_shell(ion0,sites,rng,(0.1,2.2),MIN_DIST)
            existing=sites+[ion]; parts=list(sites)+[ion]; nums=[site_z for _ in sites]+[ion_z]
            for wi in range(nw):
                w0,wz,_=embed("O",2000+ci*500+nw*30+trial*10+wi)
                w=safe_shell(w0,existing,rng,(7.5,11.5),1.70)
                existing.append(w); parts.append(w); nums.append(wz)
            pos=np.concatenate(parts); z=np.concatenate(nums)
            if nsites*site_q+ion_q!=0: raise SystemExit(f"neutrality failed {name}")
            cores.append(AtomicData(
              positions=torch.tensor(pos,dtype=torch.float32),
              atomic_numbers=torch.tensor(z,dtype=torch.long),
              forces=torch.zeros(len(z),3),energy=torch.zeros(1,1),
              charge=torch.zeros(1,1),velocities=torch.zeros(len(z),3)))
            core_meta.append({"candidate":name,"ion_charge":ion_q,"n_sites":nsites,
              "hydration":hname,"n_h2o":nw,"trial":trial,"n_atoms":len(z)})

model=AIMNet2Wrapper.from_checkpoint("aimnet2_wb97m_d3_3",device=DEVICE,compile_model=False).eval()
model.model_config.active_outputs={"energy","forces"}

core_batch=relax(make_batch(cores),model,CORE_STEPS,CORE_FMAX)
core_fmax=fmax_per_graph(core_batch)
core_accept=[bool(np.isfinite(x) and x<=ACCEPT_FMAX) for x in core_fmax]
relaxed=graph_slices(core_batch,core_meta)

systems=[]; meta=[]
for i,((core_pos,core_z),cm) in enumerate(zip(relaxed,core_meta)):
    if not core_accept[i]: continue
    rng=np.random.default_rng(1900000+i)
    for state,shell in [("near",NEAR_SHELL),("far",FAR_SHELL)]:
        cp=safe_shell(co2_0,[core_pos],rng,shell,MIN_DIST)
        pos=np.concatenate([core_pos,cp]); z=np.concatenate([core_z,co2_z])
        systems.append(AtomicData(
          positions=torch.tensor(pos,dtype=torch.float32),
          atomic_numbers=torch.tensor(z,dtype=torch.long),
          forces=torch.zeros(len(z),3),energy=torch.zeros(1,1),
          charge=torch.zeros(1,1),velocities=torch.zeros(len(z),3)))
        meta.append({**cm,"start_state":state,"n_atoms":len(z)})

if not systems: raise SystemExit("No acceptable cores")
batch=relax(make_batch(systems),model,FINAL_STEPS,FINAL_FMAX)
fmax=fmax_per_graph(batch); energies=batch.energy.reshape(-1).detach().cpu().numpy()
raw=[]
for i,m in enumerate(meta):
    fm=float(fmax[i])
    raw.append({**m,"final_fmax_eV_A":fm,"final_energy_eV":float(energies[i]),
      "accepted":bool(np.isfinite(fm) and fm<=ACCEPT_FMAX)})

summary={}
for cand in IONS:
    summary[cand]={}
    xs=[]; ys=[]
    for h,nw in HYDRATION.items():
        vals=[]; details=[]
        for t in range(N_TRIALS):
            pair=[x for x in raw if x["candidate"]==cand and x["hydration"]==h and x["trial"]==t]
            near=next((x for x in pair if x["start_state"]=="near"),None)
            far=next((x for x in pair if x["start_state"]=="far"),None)
            ok=bool(near and far and near["accepted"] and far["accepted"])
            delta=(near["final_energy_eV"]-far["final_energy_eV"]) if near and far else None
            details.append({"trial":t,"accepted_pair":ok,"delta_eV":float(delta) if delta is not None else None})
            if ok: vals.append(float(delta))
        a=np.asarray(vals,float)
        med=float(np.median(a)) if len(a) else None
        summary[cand][h]={"valid_pairs":len(vals),"median_delta_eV":med,
          "mean_delta_eV":float(np.mean(a)) if len(a) else None,"pair_details":details}
        if med is not None: xs.append(float(nw)); ys.append(med)
    dry=summary[cand]["dry_0h2o"]["median_delta_eV"]
    wet=summary[cand]["high_9h2o"]["median_delta_eV"]
    summary[cand]["hydration_shift_0_to_9_eV"]=float(wet-dry) if dry is not None and wet is not None else None
    summary[cand]["hydration_slope_eV_per_H2O"]=float(np.polyfit(xs,ys,1)[0]) if len(xs)>=3 else None
    summary[cand]["monotonic_non_decreasing"]=bool(all(ys[i+1]>=ys[i] for i in range(len(ys)-1))) if len(ys)==4 else None

usable=all(summary[c][h]["valid_pairs"]>=2 for c in IONS for h in HYDRATION)
report={
  "status":"PASS" if usable else "PARTIAL_CONVERGENCE",
  "device":DEVICE,"cuda_available":torch.cuda.is_available(),
  "test":"IRA900 Type-I robustness: carbonate control vs P2O7 candidate across 0/3/6/9 H2O, 3 trials, near/far CO2",
  "site_smiles":SITE_SMILES,
  "core_systems":len(cores),"core_accepted":int(sum(core_accept)),
  "co2_systems":len(systems),"accepted_co2_systems":int(sum(x["accepted"] for x in raw)),
  "summary":summary,"raw":raw,
  "interpretation_rule":"Within a candidate, a positive hydration shift/slope means the near CO2 state becomes less favorable as microhydration increases. Cross-ion raw total energies are not comparable; compare within-ion hydration response.",
  "decision_rule":"Promote P2O7 only if it retains a positive hydration response across the 4-point, 3-trial robustness test and is at least directionally competitive with carbonate control. Otherwise HOLD/REJECT.",
  "guardrails":[
    "Local IRA900 Type-I site model only; not full polymer or pore.",
    "Near-minus-far relaxed energy is a screening proxy, not adsorption free energy.",
    "Microhydration is not relative humidity.",
    "Different stoichiometries mean raw cross-ion energies are not directly comparable.",
    "No experimental superiority or discovery claim is allowed from this calculation alone."
  ],
  "next_gate":"If P2O7 passes robustness, cross-check the winning candidate with an independent computational method before lab recommendation."
}
(OUT/"alchemi-ira900-p2o7-robustness-v1.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))

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
FMAX_GOAL=0.08
FMAX_ACCEPT=0.30
MAX_STEPS=800
DT=0.001

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

def place_safe(fragment, existing, rng, center_radius, min_dist=MIN_INTERFRAG_A, attempts=300):
    for _ in range(attempts):
        v=rng.normal(size=3); v/=np.linalg.norm(v)
        r=rng.uniform(center_radius[0],center_radius[1])
        cand=fragment-fragment.mean(axis=0,keepdims=True)+v*r
        if all(min_cross(cand,e)>=min_dist for e in existing):
            return cand
    raise RuntimeError("safe placement failed")

site0,site_z,site_q=embed(SITE_SMILES,101)
nidx=np.where(site_z==7)[0]
if site_q!=1 or len(nidx)!=1: raise SystemExit("site sanity failed")
nidx=int(nidx[0])
sites=[place_atom(site0,nidx,t) for t in SITE_N_TARGETS]
if min_cross(sites[0],sites[1])<MIN_INTERFRAG_A: raise SystemExit("site-site overlap")
site_pos=np.concatenate(sites); site_nums=np.concatenate([site_z,site_z])

systems=[]; meta=[]; geometry_audit=[]
for ci,(name,smi) in enumerate(IONS.items()):
    ion0,ion_z,ion_q=embed(smi,500+ci)
    if 2*site_q+ion_q!=0: raise SystemExit(f"charge mismatch {name}")
    for hname,nw in HYDRATION.items():
        for trial in range(N_TRIALS):
            rng=np.random.default_rng(310000+ci*1000+nw*100+trial)
            existing=[sites[0],sites[1]]
            # Place divalent ion near the midpoint but reject all atom overlaps.
            ion=place_safe(ion0,existing,rng,(3.0,5.0),1.55)
            existing.append(ion)
            parts=[sites[0],sites[1],ion]; nums=[site_z,site_z,ion_z]
            for wi in range(nw):
                w0,wz,_=embed("O",1000+ci*100+trial*10+wi)
                w=place_safe(w0,existing,rng,(5.5,8.5),1.55)
                existing.append(w); parts.append(w); nums.append(wz)
            pos=np.concatenate(parts); z=np.concatenate(nums)
            # Full fragment-pair geometry audit.
            pairmins=[]
            for i in range(len(parts)):
                for j in range(i+1,len(parts)):
                    pairmins.append(min_cross(parts[i],parts[j]))
            mind=float(min(pairmins))
            if mind<MIN_INTERFRAG_A:
                raise SystemExit(f"geometry audit failed {name}/{hname}/{trial}: {mind}")
            geometry_audit.append({"candidate":name,"hydration":hname,"trial":trial,
                                   "min_interfragment_A":mind,"n_atoms":len(z)})
            systems.append(AtomicData(
                positions=torch.tensor(pos,dtype=torch.float32),
                atomic_numbers=torch.tensor(z,dtype=torch.long),
                forces=torch.zeros(len(z),3),
                energy=torch.zeros(1,1),
                charge=torch.zeros(1,1),
                velocities=torch.zeros(len(z),3),
            ))
            meta.append({"candidate":name,"hydration":hname,"trial":trial,"n_atoms":len(z)})

batch=Batch.from_data_list(systems,device=DEVICE)
model=AIMNet2Wrapper.from_checkpoint("aimnet2_wb97m_d3_3",device=DEVICE,compile_model=False).eval()
model.model_config.active_outputs={"energy","forces"}
opt=FIRE2(model=model,dt=DT,n_steps=MAX_STEPS,
          convergence_hook=ConvergenceHook.from_fmax(threshold=FMAX_GOAL,source_status=0,target_status=1))
for hook in model.make_neighbor_hooks():
    opt.register_hook(hook,stage=DynamicsStage.BEFORE_COMPUTE)
batch=opt.run(batch)

forces=batch.forces.detach().cpu(); bidx=batch.batch_idx.detach().cpu()
fn=forces.norm(dim=-1); fmax=torch.zeros(batch.num_graphs)
fmax.scatter_reduce_(0,bidx,fn,reduce="amax",include_self=True)
fmax=fmax.numpy()

results=[]
for i,m in enumerate(meta):
    fm=float(fmax[i])
    results.append({**m,"final_fmax_eV_A":fm,
                    "accepted":bool(np.isfinite(fm) and fm<=FMAX_ACCEPT)})

accepted=sum(r["accepted"] for r in results)
by={}
for cand in IONS:
    by[cand]={}
    for h in HYDRATION:
        rr=[r for r in results if r["candidate"]==cand and r["hydration"]==h]
        by[cand][h]={"accepted":sum(x["accepted"] for x in rr),
                     "total":len(rr),
                     "fmax":[x["final_fmax_eV_A"] for x in rr]}

report={
 "status":"PASS" if accepted>=4 else "FAIL_GEOMETRY_PREP",
 "device":DEVICE,"cuda_available":torch.cuda.is_available(),
 "test":"Gate D.1 IRA910 core-geometry preparation before CO2 energetic comparison",
 "systems":len(systems),"accepted_systems":accepted,
 "geometry_audit":geometry_audit,"results":results,"summary":by,
 "settings":{"min_interfragment_A":MIN_INTERFRAG_A,"dt":DT,"max_steps":MAX_STEPS,
             "fmax_goal_eV_A":FMAX_GOAL,"fmax_accept_eV_A":FMAX_ACCEPT},
 "decision_rule":"Do not add CO2 or compare energies unless core motifs converge reliably first.",
 "guardrails":[
   "This gate validates geometry preparation only; it does not rank HPO4 versus carbonate.",
   "Two disconnected Type-II local sites are still a local hypothesis, not full IRA910 polymer/pore.",
   "Microhydration is not relative humidity.",
   "No discovery or material validation claim is allowed from this gate."
 ]
}
(OUT/"alchemi-ira910-gate-d1-geometry-prep.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
if report["status"]!="PASS":
    raise SystemExit(2)

#!/usr/bin/env python3
import json, math, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.aimnet2 import AIMNet2Wrapper

OUT=Path("03-validation"); OUT.mkdir(exist_ok=True)
device="cuda:0" if torch.cuda.is_available() else "cpu"

# Local molecular motif inferred from the published TEMED-DQ-AER synthesis:
# benzyl-linked TEMED with both nitrogens quaternized (net +2).
CATION="[N+](C)(C)(Cc1ccccc1)CC[N+](C)(C)C"
ANIONS={
    "carbonate":"[O-]C(=O)[O-]",
    "hydrogen_phosphate":"OP(=O)([O-])[O-]",
}

def embed(smiles, seed=7):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    if m is None: raise ValueError(smiles)
    p=AllChem.ETKDGv3(); p.randomSeed=seed
    if AllChem.EmbedMolecule(m,p)!=0:
        raise RuntimeError("RDKit embed failed")
    AllChem.MMFFOptimizeMolecule(m,maxIters=500)
    conf=m.GetConformer()
    pos=np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())],dtype=np.float32)
    z=np.array([a.GetAtomicNum() for a in m.GetAtoms()],dtype=np.int64)
    return pos,z

cat_pos,cat_z=embed(CATION,11)
n_idx=np.where(cat_z==7)[0]
n_center=cat_pos[n_idx].mean(axis=0)

def complex_geometry(anion_smiles,distance):
    ap,az=embed(anion_smiles,19)
    ac=ap.mean(axis=0)
    # place anion near/far from midpoint of the two quaternary N atoms
    target=n_center+np.array([distance,0,0],dtype=np.float32)
    ap=ap-ac+target
    pos=np.concatenate([cat_pos,ap],axis=0)
    z=np.concatenate([cat_z,az],axis=0)
    return pos,z

def make_batch(pos,z):
    d=AtomicData(
        positions=torch.tensor(pos,dtype=torch.float32),
        atomic_numbers=torch.tensor(z,dtype=torch.long),
        forces=torch.zeros(len(z),3),
        energy=torch.zeros(1,1),
        charge=torch.zeros(1,1),
    )
    return Batch.from_data_list([d],device=device)

model=AIMNet2Wrapper.from_checkpoint("aimnet2_wb97m_d3_3",device=device,compile_model=False).eval()
model.model_config.active_outputs={"energy"}
results={}
for name,smi in ANIONS.items():
    vals={}
    for label,dist in [("close",4.0),("far",14.0)]:
        pos,z=complex_geometry(smi,dist)
        b=make_batch(pos,z)
        for h in model.make_neighbor_hooks():
            h(b)
        t=time.perf_counter()
        with torch.no_grad():
            o=model(b)
        e=float(o["energy"].reshape(-1)[0].detach().cpu())
        vals[label]={"energy_eV":e,"distance_A":dist,"elapsed_s":time.perf_counter()-t,"atoms":len(z)}
    vals["delta_close_minus_far_eV"]=vals["close"]["energy_eV"]-vals["far"]["energy_eV"]
    results[name]=vals

report={
 "status":"PASS",
 "device":device,
 "cuda_available":torch.cuda.is_available(),
 "method":"NVIDIA ALCHEMI AIMNet2 static close-vs-far local-motif energy comparison",
 "cation_smiles":CATION,
 "anions":ANIONS,
 "results":results,
 "interpretation_guardrails":[
   "This is a local molecular motif, not the full polymer/resin.",
   "Close-minus-far energy is a screening signal, not a rigorous binding free energy.",
   "No humidity or CO2 adsorption is modeled in this V1.",
   "If phosphorus is outside the model domain, the run should fail rather than be treated as evidence."
 ],
 "next_gate":"Add explicit water and CO2, sample multiple ion placements, and compare carbonate vs HPO4 ensembles before any lab claim."
}
(OUT/"alchemi-temed-dq-counterion-v1.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))

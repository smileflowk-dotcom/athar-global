#!/usr/bin/env python3
# Independent confirmation V2:
# GFN1-xTB (different Hamiltonian from AIMNet2) with staged geometry relaxation.
import json, math
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import FIRE
from tblite.ase import TBLite

OUT=Path("03-validation/xtb-gfn1-ira900-p2o7-crosscheck-v2.json")
SITE="C[N+](C)(C)Cc1ccccc1"
IONS={
 "carbonate":("[O-]C(=O)[O-]",-2,2),
 "p2o7":("O=P([O-])([O-])OP(=O)([O-])[O-]",-4,4),
}

def frag(smiles,seed):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    p=AllChem.ETKDGv3(); p.randomSeed=seed
    if AllChem.EmbedMolecule(m,p)!=0: raise RuntimeError("embed failed")
    AllChem.UFFOptimizeMolecule(m,maxIters=300)
    c=m.GetConformer()
    sy=[a.GetSymbol() for a in m.GetAtoms()]
    xyz=np.array([[c.GetAtomPosition(i).x,c.GetAtomPosition(i).y,c.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())])
    return sy,xyz-xyz.mean(0)

def build(name,nw,trial,state):
    smi,q,ns=IONS[name]; parts=[]
    sy,x=frag(smi,100+trial); parts.append((sy,x))
    for i in range(ns):
        a=2*math.pi*i/ns+0.21*trial
        sy,x=frag(SITE,200+10*trial+i)
        parts.append((sy,x+np.array([5*np.cos(a),5*np.sin(a),0.6*((i%2)*2-1)])))
    for i in range(nw):
        sy,x=frag("O",300+20*trial+i)
        a=2*math.pi*i/max(nw,1)+0.13*trial; r=4.2+0.4*(i%3)
        parts.append((sy,x+np.array([r*np.cos(a),r*np.sin(a),1.3*((i%3)-1)])))
    sy,x=frag("O=C=O",400+trial)
    parts.append((sy,x+np.array([3.2 if state=="near" else 9.0,0,2.2])))
    symbols=[]; coords=[]
    for sy,x in parts: symbols+=sy; coords+=x.tolist()
    return Atoms(symbols=symbols,positions=np.asarray(coords)), q+ns

def calc_one(name,nw,trial,state):
    atoms,charge=build(name,nw,trial,state)
    # GFN1-xTB is used because the prior GFN2 SCC was unstable for highly hydrated P2O7.
    atoms.calc=TBLite(method="GFN1-xTB",charge=charge,accuracy=1.5,max_iterations=1000)
    opt=FIRE(atoms,logfile=None,dt=0.05,maxstep=0.10)
    err=None
    try:
        opt.run(fmax=0.35,steps=120)
        e=float(atoms.get_potential_energy())
        f=float(np.max(np.linalg.norm(atoms.get_forces(),axis=1)))
        ok=np.isfinite(e) and np.isfinite(f)
    except Exception as ex:
        e=None; f=None; ok=False; err=str(ex)
    return dict(candidate=name,n_h2o=nw,trial=trial,state=state,energy_eV=e,
                max_force_eV_A=f,accepted=bool(ok),error=err,n_atoms=len(atoms),charge=charge)

rows=[]
for name in ["carbonate","p2o7"]:
  for nw in [0,9]:
    for trial in [0,1,2]:
      for state in ["near","far"]:
        print("RUN",name,nw,trial,state,flush=True)
        rows.append(calc_one(name,nw,trial,state))

summary={}
for name in ["carbonate","p2o7"]:
  med={}; valid={}
  for nw in [0,9]:
    ds=[]
    for t in [0,1,2]:
      a=[r for r in rows if r["candidate"]==name and r["n_h2o"]==nw and r["trial"]==t]
      n=next(r for r in a if r["state"]=="near"); f=next(r for r in a if r["state"]=="far")
      if n["accepted"] and f["accepted"]: ds.append(n["energy_eV"]-f["energy_eV"])
    med[str(nw)]=float(np.median(ds)) if ds else None
    valid[str(nw)]=len(ds)
  shift=med["9"]-med["0"] if med["9"] is not None and med["0"] is not None else None
  summary[name]=dict(median_delta_eV=med,hydration_shift_0_to_9_eV=shift,
                     positive_hydration_response=(bool(shift>0) if shift is not None else None),
                     valid_pairs=valid)

p=summary["p2o7"]["hydration_shift_0_to_9_eV"]
vp=summary["p2o7"]["valid_pairs"]
if p is not None and vp["0"]>=2 and vp["9"]>=2:
    status="PASS" if p>0 else "HOLD"
else:
    status="PARTIAL_CONVERGENCE"

result={
 "status":status,
 "method":"GFN1-xTB + FIRE staged independent semiempirical cross-check",
 "reason":"Chosen after GFN2 SCC failures on highly hydrated P2O7; independent from AIMNet2 and configured for more forgiving SCC/geometry relaxation.",
 "summary":summary,
 "rows":rows,
 "guardrails":[
   "Independent semiempirical model, not experimental proof.",
   "Local IRA900 cluster only.",
   "Microhydration is not RH.",
   "Near-minus-far energy is a screening proxy, not adsorption free energy."
 ]
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2))
print(json.dumps({"status":status,"summary":summary},indent=2))

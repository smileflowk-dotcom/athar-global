import json, math, random
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.optimize import BFGS
from tblite.ase import TBLite

OUT=Path("03-validation/xtb-ira900-p2o7-crosscheck-v1.json")
SITE="C[N+](C)(C)Cc1ccccc1"
IONS={
 "carbonate":("[O-]C(=O)[O-]",-2,2),
 "p2o7":"O=P([O-])([O-])OP(=O)([O-])[O-]"
}
IONS={"carbonate":("[O-]C(=O)[O-]",-2,2),
      "p2o7":("O=P([O-])([O-])OP(=O)([O-])[O-]",-4,4)}

def rdkit_fragment(smiles, seed):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    p=AllChem.ETKDGv3(); p.randomSeed=seed
    if AllChem.EmbedMolecule(m,p)!=0:
        raise RuntimeError("embed failed "+smiles)
    AllChem.UFFOptimizeMolecule(m,maxIters=300)
    c=m.GetConformer()
    sy=[a.GetSymbol() for a in m.GetAtoms()]
    xyz=np.array([[c.GetAtomPosition(i).x,c.GetAtomPosition(i).y,c.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())])
    xyz-=xyz.mean(0)
    return sy,xyz

def add(parts, sy, xyz, shift):
    parts.append((sy,xyz+np.asarray(shift)))

def build(name,nw,trial,state):
    smi,q,ns=IONS[name]
    rng=random.Random(9200+trial+nw*17+(0 if name=="carbonate" else 101))
    parts=[]
    sy,xyz=rdkit_fragment(smi,100+trial); add(parts,sy,xyz,[0,0,0])
    for i in range(ns):
        ang=2*math.pi*i/ns + 0.17*trial
        sy,xyz=rdkit_fragment(SITE,200+trial*10+i)
        add(parts,sy,xyz,[5.0*math.cos(ang),5.0*math.sin(ang),0.5*((i%2)*2-1)])
    for i in range(nw):
        sy,xyz=rdkit_fragment("O",300+trial*20+i)
        ang=2*math.pi*i/max(nw,1)+0.11*trial
        r=4.0+0.35*(i%3)
        add(parts,sy,xyz,[r*math.cos(ang),r*math.sin(ang),1.2*((i%3)-1)])
    sy,xyz=rdkit_fragment("O=C=O",400+trial)
    d=3.1 if state=="near" else 9.0
    add(parts,sy,xyz,[d,0,2.2])
    symbols=[]; coords=[]
    for s,x in parts: symbols+=s; coords.extend(x.tolist())
    return Atoms(symbols=symbols,positions=np.asarray(coords)),q+ns

def run_one(name,nw,trial,state):
    atoms,charge=build(name,nw,trial,state)
    atoms.calc=TBLite(method="GFN2-xTB",charge=charge,accuracy=1.0,max_iterations=250)
    opt=BFGS(atoms,logfile=None)
    try:
        opt.run(fmax=0.20,steps=80)
        converged=opt.converged()
    except Exception:
        converged=False
    e=float(atoms.get_potential_energy())
    f=float(np.max(np.linalg.norm(atoms.get_forces(),axis=1)))
    return {"candidate":name,"n_h2o":nw,"trial":trial,"state":state,
            "energy_eV":e,"max_force_eV_A":f,"converged":bool(converged),
            "n_atoms":len(atoms),"charge":charge}

rows=[]
for name in ["carbonate","p2o7"]:
  for nw in [0,9]:
    for trial in [0,1]:
      for state in ["near","far"]:
        print("RUN",name,nw,trial,state,flush=True)
        rows.append(run_one(name,nw,trial,state))

summary={}
for name in ["carbonate","p2o7"]:
  med={}
  valid={}
  for nw in [0,9]:
    ds=[]
    for t in [0,1]:
      a=[r for r in rows if r["candidate"]==name and r["n_h2o"]==nw and r["trial"]==t]
      near=next(r for r in a if r["state"]=="near")
      far=next(r for r in a if r["state"]=="far")
      ds.append(near["energy_eV"]-far["energy_eV"])
    med[str(nw)]=float(np.median(ds)); valid[str(nw)]=len(ds)
  shift=med["9"]-med["0"]
  summary[name]={"median_delta_eV":med,"hydration_shift_0_to_9_eV":shift,
                 "positive_hydration_response":bool(shift>0),"pairs":valid}

p=summary["p2o7"]["hydration_shift_0_to_9_eV"]
c=summary["carbonate"]["hydration_shift_0_to_9_eV"]
status="PASS" if p>0 else "HOLD"
result={"status":status,"method":"GFN2-xTB via tblite, independent reduced-cluster cross-check",
        "summary":summary,"rows":rows,
        "interpretation":"Positive 0->9 H2O shift means near-CO2 becomes less favorable with hydration in this reduced-cluster proxy.",
        "guardrails":["Independent semiempirical cross-check, not DFT.","Reduced local IRA900 cluster, not full resin/pore.","Microhydration is not relative humidity.","Near-minus-far is a screening proxy, not adsorption free energy.","No experimental claim."],
        "next_gate":"If P2O7 stays positive, verify phosphorus/charged-system model sensitivity and prepare a lab-facing validation protocol with carbonate control."}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,indent=2))
print(json.dumps({"status":status,"summary":summary},indent=2))

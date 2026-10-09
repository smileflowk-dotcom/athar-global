#!/usr/bin/env python3
import os,json,math
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

cand=os.environ["CANDIDATE"]
method=os.environ["METHOD"]
resin,ion=cand.split("__",1)

SITE={
 "IRA900":"C[N+](C)(C)Cc1ccccc1",
 "D201":"C[N+](C)(C)Cc1ccccc1",
 "QMPR-1":"C[N+](C)(C)CC(=O)c1ccccc1",
 "QMPR-2":"C[N+](C)(C)CC(=O)c1ccccc1",
 "QMPR-3":"C[N+](C)(C)CC(=O)c1ccccc1",
}
IONS={
 "CO3^2-":("[O-]C(=O)[O-]",-2),
 "C2O4^2-":("[O-]C(=O)C(=O)[O-]",-2),
 "citrate^3-":("O=C([O-])CC(O)(CC(=O)[O-])C(=O)[O-]",-3),
 "HCO3^-":("O=C(O)[O-]",-1),
 "B(OH)4^-":("[B-](O)(O)(O)O",-1),
 "H2PO4^-":("OP(=O)(O)[O-]",-1),
}
if resin not in SITE or ion not in IONS: raise SystemExit("unsupported candidate")

def frag(smiles,seed):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    if m is None: raise RuntimeError(smiles)
    p=AllChem.ETKDGv3(); p.randomSeed=int(seed)
    if AllChem.EmbedMolecule(m,p)!=0: raise RuntimeError("embed failed")
    try: AllChem.UFFOptimizeMolecule(m,maxIters=250)
    except Exception: pass
    c=m.GetConformer()
    sy=[a.GetSymbol() for a in m.GetAtoms()]
    xyz=np.array([[c.GetAtomPosition(i).x,c.GetAtomPosition(i).y,c.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())])
    return sy,xyz-xyz.mean(0),int(sum(a.GetFormalCharge() for a in m.GetAtoms()))

def make_system(testion,nw,trial,state):
    ion_smi,q=IONS[testion]
    syi,xi,qi=frag(ion_smi,100+trial)
    if qi!=q: raise RuntimeError(f"charge mismatch {testion}")
    nsites=abs(q)
    symbols=[]; coords=[]
    # ion at origin
    symbols+=syi; coords+=xi.tolist()
    # cationic local sites around it
    for i in range(nsites):
        sys,xs,qs=frag(SITE[resin],200+trial*10+i)
        a=2*math.pi*i/max(nsites,1)+0.17*trial
        shift=np.array([5.5*math.cos(a),5.5*math.sin(a),0.5*((i%2)*2-1)])
        symbols+=sys; coords+=(xs+shift).tolist()
    # water shell
    for i in range(nw):
        syw,xw,_=frag("O",400+trial*20+i)
        a=2*math.pi*i/max(nw,1)+0.11*trial
        rr=6.8+0.35*(i%3)
        shift=np.array([rr*math.cos(a),rr*math.sin(a),1.1*((i%3)-1)])
        symbols+=syw; coords+=(xw+shift).tolist()
    # CO2 near/far
    syc,xc,_=frag("O=C=O",800+trial)
    shift=np.array([3.6 if state=="near" else 10.5,0,2.0])
    symbols+=syc; coords+=(xc+shift).tolist()
    return symbols,np.asarray(coords,float)

def run_gfn1(testion,nw,trial,state):
    from ase import Atoms
    from ase.optimize import FIRE
    from tblite.ase import TBLite
    sy,xyz=make_system(testion,nw,trial,state)
    a=Atoms(symbols=sy,positions=xyz)
    a.calc=TBLite(method="GFN1-xTB",charge=0,accuracy=1.5,max_iterations=800)
    try:
        FIRE(a,logfile=None,dt=.05,maxstep=.10).run(fmax=.35,steps=90)
        e=float(a.get_potential_energy())
        fm=float(np.max(np.linalg.norm(a.get_forces(),axis=1)))
        return {"ok":bool(np.isfinite(e) and np.isfinite(fm)),"e":e,"fmax":fm}
    except Exception as ex:
        return {"ok":False,"error":str(ex)}

def run_aimnet(testion,nw,trial,state):
    import torch
    from nvalchemi.data import AtomicData,Batch
    from nvalchemi.dynamics.optimizers.fire2 import FIRE2
    from nvalchemi.dynamics.base import ConvergenceHook,DynamicsStage
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper
    sy,xyz=make_system(testion,nw,trial,state)
    z=np.array([Chem.GetPeriodicTable().GetAtomicNumber(s) for s in sy],dtype=np.int64)
    d=AtomicData(positions=torch.tensor(xyz,dtype=torch.float32),
      atomic_numbers=torch.tensor(z,dtype=torch.long),forces=torch.zeros(len(z),3),
      energy=torch.zeros(1,1),charge=torch.zeros(1,1),velocities=torch.zeros(len(z),3))
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    model=AIMNet2Wrapper.from_checkpoint("aimnet2_wb97m_d3_3",device=device,compile_model=False).eval()
    model.model_config.active_outputs={"energy","forces"}
    b=Batch.from_data_list([d],device=device)
    opt=FIRE2(model=model,dt=.001,n_steps=450,
      convergence_hook=ConvergenceHook.from_fmax(threshold=.12,source_status=0,target_status=1))
    for hook in model.make_neighbor_hooks(): opt.register_hook(hook,stage=DynamicsStage.BEFORE_COMPUTE)
    try:
        b=opt.run(b)
        e=float(b.energy.reshape(-1)[0].detach().cpu())
        fm=float(b.forces.norm(dim=-1).max().detach().cpu())
        return {"ok":bool(np.isfinite(e) and np.isfinite(fm) and fm<=.40),"e":e,"fmax":fm}
    except Exception as ex:
        return {"ok":False,"error":str(ex)}

runner=run_aimnet if method=="aimnet" else run_gfn1
rows=[]
for testion in ["CO3^2-",ion]:
  for nw in [0,9]:
    for trial in [0,1]:
      pair={}
      for state in ["near","far"]:
        print("RUN",method,resin,testion,nw,trial,state,flush=True)
        pair[state]=runner(testion,nw,trial,state)
      ok=pair["near"].get("ok") and pair["far"].get("ok")
      delta=(pair["near"]["e"]-pair["far"]["e"]) if ok else None
      rows.append({"ion":testion,"n_h2o":nw,"trial":trial,"ok":bool(ok),"delta_eV":delta,"detail":pair})

summary={}
for testion in ["CO3^2-",ion]:
    med={}
    counts={}
    for nw in [0,9]:
        vals=[r["delta_eV"] for r in rows if r["ion"]==testion and r["n_h2o"]==nw and r["ok"]]
        med[str(nw)]=float(np.median(vals)) if vals else None
        counts[str(nw)]=len(vals)
    shift=med["9"]-med["0"] if med["0"] is not None and med["9"] is not None else None
    summary[testion]={"median_delta_eV":med,"shift_eV":shift,"positive":(bool(shift>0) if shift is not None else None),"pairs":counts}

s=summary[ion]
status="PASS" if s["positive"] is True and s["pairs"]["0"]>=1 and s["pairs"]["9"]>=1 else ("HOLD" if s["positive"] is False else "PARTIAL")
out={"candidate":cand,"method":method,"status":status,"summary":summary,"rows":rows,
"guardrails":["local-site model only","microhydration is not RH","energy difference is a screening proxy, not adsorption free energy"]}
p=Path("03-validation/autopilot-v1"); p.mkdir(parents=True,exist_ok=True)
fn=p/(cand.replace("/","_").replace("^","").replace("(","").replace(")","")+"__"+method+".json")
fn.write_text(json.dumps(out,indent=2))
print("RESULT_FILE",fn)
print(json.dumps({"candidate":cand,"method":method,"status":status,"summary":summary},indent=2))

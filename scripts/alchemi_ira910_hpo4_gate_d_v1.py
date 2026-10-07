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

OUT = Path('03-validation')
OUT.mkdir(exist_ok=True)
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# IRA910 Type-II strong-base local-site hypothesis:
# benzyl-dimethyl-(2-hydroxyethyl)-ammonium, represented as a monocation.
# Two local sites are required to neutralize a divalent counter-ion.
SITE_SMILES = 'OCC[N+](C)(C)Cc1ccccc1'
COUNTERIONS = {
    'carbonate_control': '[O-]C(=O)[O-]',
    'hpo4_candidate': 'OP(=O)([O-])[O-]',
}
CO2 = 'O=C=O'
HYDRATION = {'dry_0h2o': 0, 'wet_6h2o': 6}
N_TRIALS = 6
SITE_SEPARATION_A = 8.0
ANION_OFFSET_A = 0.0
NEAR_A = 4.5
FAR_A = 12.0
FMAX_GOAL = 0.05
FMAX_ACCEPT = 0.20
MAX_STEPS = 450
DT = 0.003


def formal_charge(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))


def embed(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(smiles)
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError(f'Embed failed: {smiles}')
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return pos, z


def unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def place(pos, target):
    return pos - pos.mean(axis=0, keepdims=True) + np.asarray(target, dtype=np.float32)


# Motif sanity checks before any energetic calculation.
site_charge = formal_charge(SITE_SMILES)
ion_charges = {k: formal_charge(v) for k, v in COUNTERIONS.items()}
if site_charge != 1:
    raise SystemExit(f'IRA910 site model must be +1; got {site_charge}')
for name, q in ion_charges.items():
    if 2 * site_charge + q != 0:
        raise SystemExit(f'Non-neutral 2-site complex for {name}: total charge={2*site_charge+q}')

site0_pos, site_z = embed(SITE_SMILES, 101)
n_idx = np.where(site_z == 7)[0]
if len(n_idx) != 1:
    raise SystemExit('Expected exactly one N atom in Type-II local site model')

site1 = place(site0_pos, np.array([-SITE_SEPARATION_A/2, 0.0, 0.0]))
site2 = place(site0_pos, np.array([ SITE_SEPARATION_A/2, 0.0, 0.0]))
site_positions = np.concatenate([site1, site2])
site_numbers = np.concatenate([site_z, site_z])
site_n_local = np.array([n_idx[0], len(site_z)+n_idx[0]], dtype=int)
site_center = site_positions[site_n_local].mean(axis=0)
co2_pos0, co2_z = embed(CO2, 202)

motif_report = {
    'site_smiles': SITE_SMILES,
    'site_formal_charge': site_charge,
    'site_count': 2,
    'counterion_charges': ion_charges,
    'neutral_complexes': {k: (2*site_charge+q)==0 for k,q in ion_charges.items()},
    'site_separation_A': SITE_SEPARATION_A,
    'model_scope': 'two disconnected Type-II quaternary-ammonium local sites; not full IRA910 polymer/pore',
    'status': 'PASS',
}


def build(anion_smiles, trial, nwater, co2_distance, seed):
    rng = np.random.default_rng(seed)
    ap0, az = embed(anion_smiles, 500 + trial)
    ap = place(ap0, site_center + unit(rng) * ANION_OFFSET_A)
    cp = place(co2_pos0, site_center + unit(rng) * co2_distance)
    parts = [site_positions, ap, cp]
    nums = [site_numbers, az, co2_z]
    for i in range(nwater):
        wp0, wz = embed('O', 1000 + trial*20 + i)
        radius = 4.2 + 0.45*i
        parts.append(place(wp0, site_center + unit(rng) * radius))
        nums.append(wz)
    pos = np.concatenate(parts)
    z = np.concatenate(nums)
    co2_start = len(site_numbers) + len(az)
    co2_c_local = int(np.where(co2_z == 6)[0][0])
    return pos, z, co2_start + co2_c_local

systems, meta = [], []
for ci, (name, smi) in enumerate(COUNTERIONS.items()):
    for hname, nw in HYDRATION.items():
        for trial in range(N_TRIALS):
            seed = 220000 + ci*1000 + nw*100 + trial
            for state, dist in [('near', NEAR_A), ('far', FAR_A)]:
                pos, z, co2_c_idx = build(smi, trial, nw, dist, seed)
                systems.append(AtomicData(
                    positions=torch.tensor(pos, dtype=torch.float32),
                    atomic_numbers=torch.tensor(z, dtype=torch.long),
                    forces=torch.zeros(len(z), 3),
                    energy=torch.zeros(1, 1),
                    charge=torch.zeros(1, 1),
                    velocities=torch.zeros(len(z), 3),
                ))
                meta.append({
                    'candidate': name, 'hydration': hname, 'n_water': nw,
                    'trial': trial, 'start_state': state,
                    'co2_c_local_idx': co2_c_idx, 'n_atoms': len(z),
                })

batch = Batch.from_data_list(systems, device=DEVICE)
model = AIMNet2Wrapper.from_checkpoint('aimnet2_wb97m_d3_3', device=DEVICE, compile_model=False).eval()
model.model_config.active_outputs = {'energy', 'forces'}
opt = FIRE2(
    model=model, dt=DT, n_steps=MAX_STEPS,
    convergence_hook=ConvergenceHook.from_fmax(
        threshold=FMAX_GOAL, source_status=0, target_status=1
    ),
)
for hook in model.make_neighbor_hooks():
    opt.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
batch = opt.run(batch)

energies = batch.energy.reshape(-1).detach().cpu().numpy()
forces = batch.forces.detach().cpu()
positions = batch.positions.detach().cpu()
batch_idx = batch.batch_idx.detach().cpu()
force_norms = forces.norm(dim=-1)
fmax = torch.zeros(batch.num_graphs)
fmax.scatter_reduce_(0, batch_idx, force_norms, reduce='amax', include_self=True)
fmax = fmax.numpy()

offsets, cursor = [], 0
for m in meta:
    offsets.append(cursor)
    cursor += m['n_atoms']

raw=[]
for i,m in enumerate(meta):
    off=offsets[i]
    cglob=off+m['co2_c_local_idx']
    nglob=off+site_n_local
    center=positions[nglob].mean(dim=0)
    dist=float(torch.linalg.norm(positions[cglob]-center))
    fm=float(fmax[i])
    raw.append({**m,'final_energy_eV':float(energies[i]),'final_fmax_eV_A':fm,
                'accepted':bool(np.isfinite(fm) and fm<=FMAX_ACCEPT),
                'final_co2C_to_sitecenter_A':dist})

summary={}
for cand in COUNTERIONS:
    summary[cand]={}
    for hname in HYDRATION:
        vals=[]; details=[]
        for trial in range(N_TRIALS):
            near=next(x for x in raw if x['candidate']==cand and x['hydration']==hname and x['trial']==trial and x['start_state']=='near')
            far=next(x for x in raw if x['candidate']==cand and x['hydration']==hname and x['trial']==trial and x['start_state']=='far')
            ok=near['accepted'] and far['accepted']
            delta=near['final_energy_eV']-far['final_energy_eV']
            details.append({'trial':trial,'accepted_pair':ok,'delta_eV':float(delta),
                            'near_fmax':near['final_fmax_eV_A'],'far_fmax':far['final_fmax_eV_A']})
            if ok: vals.append(delta)
        arr=np.asarray(vals,float)
        summary[cand][hname]={
            'valid_pairs':len(vals),'pair_details':details,
            'median_delta_eV':float(np.median(arr)) if len(arr) else None,
            'mean_delta_eV':float(arr.mean()) if len(arr) else None,
            'std_delta_eV':float(arr.std(ddof=1)) if len(arr)>1 else None,
        }
    dry=summary[cand]['dry_0h2o']['median_delta_eV']
    wet=summary[cand]['wet_6h2o']['median_delta_eV']
    summary[cand]['hydration_shift_eV']=float(wet-dry) if dry is not None and wet is not None else None
    dp=summary[cand]['dry_0h2o']['valid_pairs']; wp=summary[cand]['wet_6h2o']['valid_pairs']
    summary[cand]['scientific_status']=(
        'robust_screening_signal' if dp>=4 and wp>=4 else
        'usable_screening_signal' if dp>=2 and wp>=2 else
        'provisional_insufficient_pairs' if dp>=1 and wp>=1 else
        'inconclusive_due_to_convergence'
    )

report={
    'status':'PASS','device':DEVICE,'cuda_available':torch.cuda.is_available(),
    'test':'Gate D targeted IRA910 Type-II local-motif HPO4 vs carbonate ALCHEMI screen',
    'motif_sanity':motif_report,
    'method':'NVIDIA ALCHEMI AIMNet2 + FIRE2; two-site neutral local motif',
    'systems':len(systems),'accepted_systems':sum(x['accepted'] for x in raw),
    'fmax_goal_eV_A':FMAX_GOAL,'fmax_accept_eV_A':FMAX_ACCEPT,
    'summary':summary,'raw':raw,
    'guardrails':[
        'The Type-II site is a chemically motivated local hypothesis, not a full IRA910 polymer model.',
        'Two monocationic sites are used so each divalent counter-ion complex is neutral.',
        'Dry/wet microhydration is a proxy, not relative humidity.',
        'Energy deltas are screening proxies, not adsorption free energies.',
        'Only converged near/far pairs enter medians.',
        'No candidate is validated by this atomistic screen.',
        'Independent computational cross-check required before lab promotion.'
    ],
    'decision_rule':'HPO4 may advance only if convergence is usable/robust and its hydration-response signal is not clearly worse than carbonate across accepted pairs.'
}
(OUT/'alchemi-ira910-hpo4-gate-d-v1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

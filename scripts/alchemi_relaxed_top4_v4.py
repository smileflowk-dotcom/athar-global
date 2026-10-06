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

OUT = Path("03-validation")
OUT.mkdir(exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

CATION = "[N+](C)(C)(Cc1ccccc1)CC[N+](C)(C)C"
CANDIDATES = {
    "carbonate_control": "[O-]C(=O)[O-]",
    "sulfite": "[O-]S(=O)[O-]",
    "succinate": "[O-]C(=O)CCC(=O)[O-]",
    "malonate": "[O-]C(=O)CC(=O)[O-]",
}
CO2 = "O=C=O"
HYDRATION = {"low_2h2o": 2, "high_6h2o": 6}
N_TRIALS = 3
NEAR_A = 4.5
FAR_A = 12.0

def embed(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(smiles)
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError(f"Embed failed: {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return pos, z

def unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)

def centered(pos):
    return pos - pos.mean(axis=0, keepdims=True)

def placed(pos, target):
    return centered(pos) + np.asarray(target, dtype=np.float32)

cat_pos, cat_z = embed(CATION, 11)
n_local = np.where(cat_z == 7)[0]
site = cat_pos[n_local].mean(axis=0)
co2_pos0, co2_z = embed(CO2, 23)

def build(smiles, trial, nwater, co2_distance, seed):
    rng = np.random.default_rng(seed)
    ap0, az = embed(smiles, 100 + trial)
    ap = placed(ap0, site + unit(rng) * 5.5)
    cp = placed(co2_pos0, site + unit(rng) * co2_distance)
    parts = [cat_pos, ap, cp]
    nums = [cat_z, az, co2_z]
    for i in range(nwater):
        wp0, wz = embed("O", 1000 + trial * 20 + i)
        parts.append(placed(wp0, site + unit(rng) * (4.5 + 0.45*i)))
        nums.append(wz)
    pos = np.concatenate(parts)
    z = np.concatenate(nums)
    # carbon index of CO2 in the combined system
    co2_start = len(cat_z) + len(az)
    co2_c_local = int(np.where(co2_z == 6)[0][0])
    return pos, z, co2_start + co2_c_local

systems = []
meta = []
for ci, (name, smi) in enumerate(CANDIDATES.items()):
    for hname, nw in HYDRATION.items():
        for trial in range(N_TRIALS):
            seed = 91000 + ci*1000 + nw*100 + trial
            for state, dist in [("near", NEAR_A), ("far", FAR_A)]:
                pos, z, co2_c_idx = build(smi, trial, nw, dist, seed)
                data = AtomicData(
                    positions=torch.tensor(pos, dtype=torch.float32),
                    atomic_numbers=torch.tensor(z, dtype=torch.long),
                    forces=torch.zeros(len(z), 3),
                    energy=torch.zeros(1, 1),
                    charge=torch.zeros(1, 1),
                    velocities=torch.zeros(len(z), 3),
                )
                systems.append(data)
                meta.append({
                    "candidate": name,
                    "hydration": hname,
                    "n_water": nw,
                    "trial": trial,
                    "start_state": state,
                    "co2_c_local_idx": co2_c_idx,
                    "n_atoms": len(z),
                })

batch = Batch.from_data_list(systems, device=device)
model = AIMNet2Wrapper.from_checkpoint(
    "aimnet2_wb97m_d3_3", device=device, compile_model=False
).eval()
model.model_config.active_outputs = {"energy", "forces"}

opt = FIRE2(
    model=model,
    dt=0.01,
    n_steps=120,
    convergence_hook=ConvergenceHook.from_fmax(
        threshold=0.10,
        source_status=0,
        target_status=1,
    ),
)
for hook in model.make_neighbor_hooks():
    opt.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)

batch = opt.run(batch)

energies = batch.energy.reshape(-1).detach().cpu().numpy()
forces = batch.forces.detach().cpu()
positions = batch.positions.detach().cpu()
batch_idx = batch.batch_idx.detach().cpu()

# per-system fmax
force_norms = forces.norm(dim=-1)
fmax = torch.zeros(batch.num_graphs)
fmax.scatter_reduce_(0, batch_idx, force_norms, reduce="amax", include_self=True)
fmax = fmax.numpy()

# graph atom offsets for extracting local CO2 carbon position
offsets = []
cursor = 0
for m in meta:
    offsets.append(cursor)
    cursor += m["n_atoms"]

raw = []
for i, m in enumerate(meta):
    off = offsets[i]
    co2_c_global = off + m["co2_c_local_idx"]
    # use midpoint of first two N atoms of local cation after relaxation
    local_n = off + n_local
    relaxed_site = positions[local_n].mean(dim=0)
    d = float(torch.linalg.norm(positions[co2_c_global] - relaxed_site))
    raw.append({
        **m,
        "final_energy_eV": float(energies[i]),
        "final_fmax_eV_A": float(fmax[i]),
        "final_co2C_to_site_A": d,
    })

# paired near-vs-far relaxed energy differences, then humidity shift
summary = {}
for cand in CANDIDATES:
    summary[cand] = {}
    for hname in HYDRATION:
        deltas = []
        dist_changes = []
        for trial in range(N_TRIALS):
            near = next(x for x in raw if x["candidate"]==cand and x["hydration"]==hname and x["trial"]==trial and x["start_state"]=="near")
            far = next(x for x in raw if x["candidate"]==cand and x["hydration"]==hname and x["trial"]==trial and x["start_state"]=="far")
            deltas.append(near["final_energy_eV"] - far["final_energy_eV"])
            dist_changes.append(near["final_co2C_to_site_A"])
        a = np.asarray(deltas, float)
        summary[cand][hname] = {
            "paired_relaxed_delta_eV": [float(v) for v in a],
            "median_delta_eV": float(np.median(a)),
            "mean_delta_eV": float(a.mean()),
            "std_delta_eV": float(a.std(ddof=1)) if len(a)>1 else 0.0,
            "median_final_near_distance_A": float(np.median(dist_changes)),
        }
    lo = summary[cand]["low_2h2o"]["median_delta_eV"]
    hi = summary[cand]["high_6h2o"]["median_delta_eV"]
    summary[cand]["moisture_shift_relaxed_eV"] = float(hi - lo)

ranking = sorted(
    [
        {
            "candidate": c,
            "low_median_delta_eV": summary[c]["low_2h2o"]["median_delta_eV"],
            "high_median_delta_eV": summary[c]["high_6h2o"]["median_delta_eV"],
            "moisture_shift_relaxed_eV": summary[c]["moisture_shift_relaxed_eV"],
        }
        for c in CANDIDATES
    ],
    key=lambda x: x["moisture_shift_relaxed_eV"],
    reverse=True,
)

report = {
    "status": "PASS",
    "device": device,
    "cuda_available": torch.cuda.is_available(),
    "method": "ALCHEMI AIMNet2 FIRE2 geometry-relaxed paired CO2 near/far comparison",
    "optimizer_steps": int(opt.step_count),
    "systems": len(systems),
    "candidates": list(CANDIDATES),
    "summary": summary,
    "ranking": ranking,
    "raw": raw,
    "guardrails": [
        "Geometry-relaxed local molecular motifs, not a full polymer/pore or adsorption isotherm.",
        "2 and 6 explicit waters are microhydration proxies, not RH values.",
        "Paired relaxed energy differences are screening proxies, not rigorous binding or free energies.",
        "CPU runtime; ALCHEMI/AIMNet2 execution is valid but not GPU acceleration.",
        "A second computational method and literature/synthetic plausibility checks are required before lab prioritization."
    ],
    "next_gate": "Cross-check the top relaxed candidates with an independent method and verify novelty/synthetic plausibility before expanding the candidate family."
}
(OUT/"alchemi-relaxed-top4-v4.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

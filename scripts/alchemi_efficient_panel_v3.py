#!/usr/bin/env python3
import json, time
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.aimnet2 import AIMNet2Wrapper
from nvalchemi.neighbors import compute_neighbors

OUT = Path("03-validation")
OUT.mkdir(exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Fixed local DQ-AER cation motif (+2) for a controlled first screening.
CATION = "[N+](C)(C)(Cc1ccccc1)CC[N+](C)(C)C"

# 12 chemically distinct dianions. Each is paired with two microhydration
# environments (2 and 6 waters) => 24 candidate conditions.
# This is a screening panel, not a claim of synthetic feasibility.
ANIONS = {
    "carbonate": "[O-]C(=O)[O-]",
    "hydrogen_phosphate": "OP(=O)([O-])[O-]",
    "sulfate": "[O-]S(=O)(=O)[O-]",
    "sulfite": "[O-]S(=O)[O-]",
    "oxalate": "[O-]C(=O)C(=O)[O-]",
    "malonate": "[O-]C(=O)CC(=O)[O-]",
    "succinate": "[O-]C(=O)CCC(=O)[O-]",
    "maleate": "[O-]C(=O)/C=C\\C(=O)[O-]",
    "fumarate": "[O-]C(=O)/C=C/C(=O)[O-]",
    "glutarate": "[O-]C(=O)CCCC(=O)[O-]",
    "adipate": "[O-]C(=O)CCCCC(=O)[O-]",
    "tartrate": "[O-]C(=O)C(O)C(O)C(=O)[O-]",
}

CO2 = "O=C=O"
HYDRATION_LEVELS = {"low_hydration_2h2o": 2, "high_hydration_6h2o": 6}
N_TRIALS = 6
NEAR_A = 4.5
FAR_A = 14.0

def embed(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError(f"RDKit embed failed for {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return pos, z

def unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)

def place(pos, target):
    return pos - pos.mean(axis=0, keepdims=True) + np.asarray(target, dtype=np.float32)

cat_pos, cat_z = embed(CATION, 11)
site = cat_pos[np.where(cat_z == 7)[0]].mean(axis=0)
co2_pos0, co2_z = embed(CO2, 23)

def build(anion_smiles, trial, n_water, co2_distance, seed_base):
    rng = np.random.default_rng(seed_base)
    ap0, az = embed(anion_smiles, 100 + trial)
    ap = place(ap0, site + unit(rng) * 5.5)
    cp = place(co2_pos0, site + unit(rng) * co2_distance)
    pos_parts = [cat_pos, ap, cp]
    z_parts = [cat_z, az, co2_z]
    for i in range(n_water):
        wp0, wz = embed("O", 1000 + trial * 20 + i)
        radius = 4.5 + 0.45 * i
        pos_parts.append(place(wp0, site + unit(rng) * radius))
        z_parts.append(wz)
    return np.concatenate(pos_parts), np.concatenate(z_parts)

def make_batch(pos, z):
    d = AtomicData(
        positions=torch.tensor(pos, dtype=torch.float32),
        atomic_numbers=torch.tensor(z, dtype=torch.long),
        forces=torch.zeros(len(z), 3),
        energy=torch.zeros(1, 1),
        charge=torch.zeros(1, 1),
    )
    return Batch.from_data_list([d], device=device)

model = AIMNet2Wrapper.from_checkpoint(
    "aimnet2_wb97m_d3_3", device=device, compile_model=False
).eval()
model.model_config.active_outputs = {"energy"}

def energy(pos, z):
    b = make_batch(pos, z)
    compute_neighbors(b, config=model.model_config.neighbor_config)
    with torch.no_grad():
        out = model(b)
    return float(out["energy"].reshape(-1)[0].detach().cpu())

rows = []
for ai, (anion_name, anion_smiles) in enumerate(ANIONS.items()):
    for hydration_name, n_water in HYDRATION_LEVELS.items():
        deltas = []
        for trial in range(N_TRIALS):
            seed = 70000 + ai * 1000 + n_water * 100 + trial
            near_pos, near_z = build(anion_smiles, trial, n_water, NEAR_A, seed)
            far_pos, far_z = build(anion_smiles, trial, n_water, FAR_A, seed)
            e_near = energy(near_pos, near_z)
            e_far = energy(far_pos, far_z)
            deltas.append(e_near - e_far)
        a = np.array(deltas, dtype=float)
        rows.append({
            "candidate_id": f"{anion_name}__{hydration_name}",
            "anion": anion_name,
            "hydration": hydration_name,
            "n_water": n_water,
            "median_delta_eV": float(np.median(a)),
            "mean_delta_eV": float(a.mean()),
            "std_delta_eV": float(a.std(ddof=1)),
            "min_delta_eV": float(a.min()),
            "max_delta_eV": float(a.max()),
            "n_trials": N_TRIALS,
        })

# Screening score: prefer a lower CO2 near-vs-far energy at low hydration
# and a higher value at high hydration for the same anion (release on wetting).
by_anion = {}
for name in ANIONS:
    lo = next(r for r in rows if r["anion"] == name and r["n_water"] == 2)
    hi = next(r for r in rows if r["anion"] == name and r["n_water"] == 6)
    shift = hi["median_delta_eV"] - lo["median_delta_eV"]
    by_anion[name] = {
        "low_hydration_median_delta_eV": lo["median_delta_eV"],
        "high_hydration_median_delta_eV": hi["median_delta_eV"],
        "moisture_shift_eV": float(shift),
        "screen_score": float(shift - max(lo["median_delta_eV"], 0.0)),
    }

ranking = sorted(
    [{"anion": k, **v} for k, v in by_anion.items()],
    key=lambda x: x["screen_score"],
    reverse=True,
)

report = {
    "status": "PASS",
    "device": device,
    "cuda_available": torch.cuda.is_available(),
    "method": "ALCHEMI AIMNet2 efficient 24-condition screen: 12 dianions x 2 hydration levels, paired CO2 near/far, 6 trials",
    "candidate_conditions": len(rows),
    "unique_anions": len(ANIONS),
    "rows": rows,
    "ranking": ranking,
    "guardrails": [
        "Screening only: local DQ-AER molecular motif, not full polymer or pore.",
        "2 and 6 explicit waters are microhydration states, not relative humidity values.",
        "Static geometries, no free-energy calculation and no experimental adsorption capacity.",
        "Ranking is heuristic and must be confirmed by geometry relaxation and a second method.",
        "Chemical/synthetic feasibility and literature novelty are separate filters after this computational screen."
    ],
    "next_gate": "Take top 5 computational signals, verify literature/novelty and synthetic plausibility, then run relaxed multi-geometry water+CO2 calculations."
}
(OUT / "alchemi-efficient-panel-v3.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

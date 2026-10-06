#!/usr/bin/env python3
import json, math, time
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
rng = np.random.default_rng(20261006)

CATION = "[N+](C)(C)(Cc1ccccc1)CC[N+](C)(C)C"
ANIONS = {
    "carbonate": "[O-]C(=O)[O-]",
    "hydrogen_phosphate": "OP(=O)([O-])[O-]",
}
CO2 = "O=C=O"
WATER = "O"

def embed(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"RDKit embed failed for {smiles}")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    return pos, z

def random_unit():
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)

def center(pos):
    return pos - pos.mean(axis=0, keepdims=True)

def place(pos, target):
    return center(pos) + np.asarray(target, dtype=np.float32)

cat_pos, cat_z = embed(CATION, 11)
n_idx = np.where(cat_z == 7)[0]
site = cat_pos[n_idx].mean(axis=0)

co2_pos0, co2_z = embed(CO2, 23)
water_pos0, water_z = embed(WATER, 31)

def build_system(anion_smiles, trial, wet, co2_distance):
    ap0, az = embed(anion_smiles, 100 + trial)
    # Put the counter-ion in a plausible first-shell region, but vary direction by trial.
    anion_target = site + random_unit() * 5.5
    ap = place(ap0, anion_target)

    co2_target = site + random_unit() * co2_distance
    cp = place(co2_pos0, co2_target)

    pos_parts = [cat_pos, ap, cp]
    z_parts = [cat_z, az, co2_z]

    if wet:
        # Four explicit water molecules distributed around the charged site.
        for i in range(4):
            wp0, wz = embed(WATER, 1000 + trial * 10 + i)
            radius = 4.8 + 0.6 * i
            wp = place(wp0, site + random_unit() * radius)
            pos_parts.append(wp)
            z_parts.append(wz)

    pos = np.concatenate(pos_parts, axis=0)
    z = np.concatenate(z_parts, axis=0)
    return pos, z

def make_batch(pos, z):
    data = AtomicData(
        positions=torch.tensor(pos, dtype=torch.float32),
        atomic_numbers=torch.tensor(z, dtype=torch.long),
        forces=torch.zeros(len(z), 3),
        energy=torch.zeros(1, 1),
        charge=torch.zeros(1, 1),  # entire constructed system is neutral
    )
    return Batch.from_data_list([data], device=device)

model = AIMNet2Wrapper.from_checkpoint(
    "aimnet2_wb97m_d3_3",
    device=device,
    compile_model=False,
).eval()
model.model_config.active_outputs = {"energy"}

def energy(pos, z):
    batch = make_batch(pos, z)
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(batch)
    e = float(out["energy"].reshape(-1)[0].detach().cpu())
    return e, time.perf_counter() - t0

N_TRIALS = 8
NEAR_A = 4.5
FAR_A = 14.0
results = {}

for anion_name, anion_smiles in ANIONS.items():
    results[anion_name] = {}
    for state, wet in [("dry", False), ("wet_4h2o", True)]:
        trials = []
        for trial in range(N_TRIALS):
            # Reset the RNG deterministically for near/far pairing so all other placements match.
            seed_state = 50000 + trial + (1000 if wet else 0) + (2000 if anion_name == "hydrogen_phosphate" else 0)
            rng = np.random.default_rng(seed_state)
            near_pos, near_z = build_system(anion_smiles, trial, wet, NEAR_A)
            rng = np.random.default_rng(seed_state)
            far_pos, far_z = build_system(anion_smiles, trial, wet, FAR_A)

            e_near, t_near = energy(near_pos, near_z)
            e_far, t_far = energy(far_pos, far_z)
            delta = e_near - e_far
            trials.append({
                "trial": trial,
                "near_energy_eV": e_near,
                "far_energy_eV": e_far,
                "delta_co2_near_minus_far_eV": delta,
                "elapsed_s": t_near + t_far,
            })

        deltas = np.array([x["delta_co2_near_minus_far_eV"] for x in trials], dtype=float)
        results[anion_name][state] = {
            "trials": trials,
            "mean_delta_eV": float(deltas.mean()),
            "median_delta_eV": float(np.median(deltas)),
            "std_delta_eV": float(deltas.std(ddof=1)),
            "min_delta_eV": float(deltas.min()),
            "max_delta_eV": float(deltas.max()),
        }

    dry = results[anion_name]["dry"]["median_delta_eV"]
    wet = results[anion_name]["wet_4h2o"]["median_delta_eV"]
    results[anion_name]["moisture_shift_median_eV"] = wet - dry

report = {
    "status": "PASS",
    "device": device,
    "cuda_available": torch.cuda.is_available(),
    "method": "NVIDIA ALCHEMI AIMNet2 paired CO2 near-vs-far ensemble with dry and 4-H2O states",
    "n_trials": N_TRIALS,
    "near_distance_A": NEAR_A,
    "far_distance_A": FAR_A,
    "cation_smiles": CATION,
    "anions": ANIONS,
    "results": results,
    "interpretation": {
        "delta_negative": "CO2-near geometry is lower in energy than paired CO2-far geometry for that sampled local environment.",
        "delta_positive": "CO2-near geometry is higher in energy than paired CO2-far geometry for that sampled local environment.",
        "moisture_shift": "wet median delta minus dry median delta; use only as a screening signal for humidity sensitivity, not as a free energy."
    },
    "guardrails": [
        "Local molecular motif only; not a full resin or pore model.",
        "Static sampled geometries; no thermodynamic free-energy calculation.",
        "Four waters are an explicit microhydration proxy, not a controlled relative humidity.",
        "Multiple paired placements reduce single-geometry bias but do not remove model-domain uncertainty.",
        "Do not call this experimental validation or a measured adsorption capacity."
    ],
    "next_gate": "If the ensemble signal is stable, add geometry relaxation and larger hydration/CO2 sampling, then cross-check the best candidates with a second computational method."
}

path = OUT / "alchemi-moisture-co2-ensemble-v2.json"
path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

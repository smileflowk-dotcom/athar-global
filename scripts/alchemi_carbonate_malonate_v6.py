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
    "malonate": "[O-]C(=O)CC(=O)[O-]",
}
CO2 = "O=C=O"
HYDRATION = {"low_2h2o": 2, "high_6h2o": 6}
N_TRIALS = 8
NEAR_A = 4.5
FAR_A = 12.0
FMAX_GOAL = 0.05
FMAX_ACCEPT = 0.20
MAX_STEPS = 400
DT = 0.003

def embed(smiles, seed):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError(smiles)
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError(f"Embed failed: {smiles}")
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

cat_pos, cat_z = embed(CATION, 11)
n_local = np.where(cat_z == 7)[0]
site = cat_pos[n_local].mean(axis=0)
co2_pos0, co2_z = embed(CO2, 23)

def build(smiles, trial, nwater, co2_distance, seed):
    rng = np.random.default_rng(seed)
    ap0, az = embed(smiles, 100 + trial)
    ap = place(ap0, site + unit(rng) * 5.5)
    cp = place(co2_pos0, site + unit(rng) * co2_distance)
    parts = [cat_pos, ap, cp]
    nums = [cat_z, az, co2_z]
    for i in range(nwater):
        wp0, wz = embed("O", 1000 + trial * 20 + i)
        parts.append(place(wp0, site + unit(rng) * (4.8 + 0.55*i)))
        nums.append(wz)
    pos = np.concatenate(parts)
    z = np.concatenate(nums)
    co2_start = len(cat_z) + len(az)
    co2_c_local = int(np.where(co2_z == 6)[0][0])
    return pos, z, co2_start + co2_c_local

systems, meta = [], []
for ci, (name, smi) in enumerate(CANDIDATES.items()):
    for hname, nw in HYDRATION.items():
        for trial in range(N_TRIALS):
            seed = 120000 + ci*1000 + nw*100 + trial
            for state, dist in [("near", NEAR_A), ("far", FAR_A)]:
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
                    "candidate": name, "hydration": hname, "n_water": nw,
                    "trial": trial, "start_state": state,
                    "co2_c_local_idx": co2_c_idx, "n_atoms": len(z),
                })

batch = Batch.from_data_list(systems, device=device)
model = AIMNet2Wrapper.from_checkpoint(
    "aimnet2_wb97m_d3_3", device=device, compile_model=False
).eval()
model.model_config.active_outputs = {"energy", "forces"}

opt = FIRE2(
    model=model,
    dt=DT,
    n_steps=MAX_STEPS,
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
fmax.scatter_reduce_(0, batch_idx, force_norms, reduce="amax", include_self=True)
fmax = fmax.numpy()

offsets, cursor = [], 0
for m in meta:
    offsets.append(cursor)
    cursor += m["n_atoms"]

raw = []
for i, m in enumerate(meta):
    off = offsets[i]
    co2_c_global = off + m["co2_c_local_idx"]
    local_n = off + n_local
    relaxed_site = positions[local_n].mean(dim=0)
    d = float(torch.linalg.norm(positions[co2_c_global] - relaxed_site))
    fm = float(fmax[i])
    raw.append({
        **m,
        "final_energy_eV": float(energies[i]),
        "final_fmax_eV_A": fm,
        "accepted": bool(np.isfinite(fm) and fm <= FMAX_ACCEPT),
        "final_co2C_to_site_A": d,
    })

summary = {}
for cand in CANDIDATES:
    summary[cand] = {}
    for hname in HYDRATION:
        deltas, pair_details = [], []
        for trial in range(N_TRIALS):
            near = next(x for x in raw if x["candidate"]==cand and x["hydration"]==hname and x["trial"]==trial and x["start_state"]=="near")
            far = next(x for x in raw if x["candidate"]==cand and x["hydration"]==hname and x["trial"]==trial and x["start_state"]=="far")
            accepted_pair = near["accepted"] and far["accepted"]
            delta = near["final_energy_eV"] - far["final_energy_eV"]
            pair_details.append({
                "trial": trial,
                "accepted_pair": accepted_pair,
                "delta_eV": float(delta),
                "near_fmax": near["final_fmax_eV_A"],
                "far_fmax": far["final_fmax_eV_A"],
            })
            if accepted_pair:
                deltas.append(delta)
        if deltas:
            a = np.asarray(deltas, float)
            summary[cand][hname] = {
                "valid_pairs": len(deltas),
                "pair_details": pair_details,
                "median_delta_eV": float(np.median(a)),
                "mean_delta_eV": float(a.mean()),
                "std_delta_eV": float(a.std(ddof=1)) if len(a)>1 else None,
            }
        else:
            summary[cand][hname] = {
                "valid_pairs": 0,
                "pair_details": pair_details,
                "median_delta_eV": None,
                "mean_delta_eV": None,
                "std_delta_eV": None,
            }

    lo = summary[cand]["low_2h2o"]["median_delta_eV"]
    hi = summary[cand]["high_6h2o"]["median_delta_eV"]
    low_pairs = summary[cand]["low_2h2o"]["valid_pairs"]
    high_pairs = summary[cand]["high_6h2o"]["valid_pairs"]

    if lo is not None and hi is not None:
        summary[cand]["moisture_shift_relaxed_eV"] = float(hi - lo)
    else:
        summary[cand]["moisture_shift_relaxed_eV"] = None

    if low_pairs >= 5 and high_pairs >= 5:
        summary[cand]["scientific_status"] = "robust_screening_signal"
    elif low_pairs >= 2 and high_pairs >= 2:
        summary[cand]["scientific_status"] = "usable_screening_signal"
    elif low_pairs >= 1 and high_pairs >= 1:
        summary[cand]["scientific_status"] = "provisional_insufficient_pairs"
    else:
        summary[cand]["scientific_status"] = "inconclusive_due_to_convergence"

ranking = sorted(
    [
        {
            "candidate": c,
            "low_median_delta_eV": summary[c]["low_2h2o"]["median_delta_eV"],
            "high_median_delta_eV": summary[c]["high_6h2o"]["median_delta_eV"],
            "moisture_shift_relaxed_eV": summary[c]["moisture_shift_relaxed_eV"],
            "low_valid_pairs": summary[c]["low_2h2o"]["valid_pairs"],
            "high_valid_pairs": summary[c]["high_6h2o"]["valid_pairs"],
            "scientific_status": summary[c]["scientific_status"],
        }
        for c in CANDIDATES
    ],
    key=lambda x: (-1e99 if x["moisture_shift_relaxed_eV"] is None else x["moisture_shift_relaxed_eV"]),
    reverse=True,
)

accepted_systems = sum(1 for x in raw if x["accepted"])
report = {
    "status": "PASS",
    "device": device,
    "cuda_available": torch.cuda.is_available(),
    "method": "ALCHEMI AIMNet2 FIRE2 focused carbonate-vs-malonate convergence-filtered validation",
    "optimizer_steps": int(opt.step_count),
    "dt": DT,
    "fmax_goal_eV_A": FMAX_GOAL,
    "fmax_accept_eV_A": FMAX_ACCEPT,
    "systems": len(systems),
    "accepted_systems": accepted_systems,
    "acceptance_rate": accepted_systems / len(systems),
    "summary": summary,
    "ranking": ranking,
    "raw": raw,
    "guardrails": [
        "Only pairs with both near and far final fmax <= acceptance threshold enter the ranking.",
        "Local molecular motif only, not full polymer/pore or adsorption isotherm.",
        "Microhydration states are proxies, not relative humidity values.",
        "Relaxed energy differences are screening proxies, not rigorous free energies.",
        "CPU execution; no GPU acceleration claim.",
        "Independent computational cross-check required before any lab prioritization."
    ],
    "next_gate": "Promote malonate only if it reaches at least 5 valid near/far pairs in BOTH hydration states and remains competitive with carbonate. Then cross-check with an independent computational method."
}
(OUT/"alchemi-carbonate-malonate-v6.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))

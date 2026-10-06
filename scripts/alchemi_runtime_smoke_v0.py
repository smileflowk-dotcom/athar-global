#!/usr/bin/env python3
import json, platform, sys, time
from pathlib import Path
import torch
import nvalchemi
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.models.lj import LennardJonesModelWrapper

outdir = Path("03-validation")
outdir.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
env = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nvalchemi": getattr(nvalchemi, "__version__", "unknown"),
}

# Official ALCHEMI-style geometry optimization smoke test.
model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
hooks = model.make_neighbor_hooks()
positions = torch.tensor([
    [0.0, 0.0, 0.0],
    [4.4, 0.0, 0.0],
    [0.0, 4.4, 0.0],
    [0.0, 0.0, 4.4],
], dtype=torch.float32, device=device)
data = AtomicData(
    positions=positions,
    atomic_numbers=torch.full((4,), 18, dtype=torch.long, device=device),
    forces=torch.zeros(4,3, device=device),
    energy=torch.zeros(1,1, device=device),
    velocities=torch.zeros(4,3, device=device),
)
batch = Batch.from_data_list([data])
opt = FIRE2(
    model=model,
    dt=0.05,
    n_steps=80,
    convergence_hook=ConvergenceHook(criteria=[{
        "key":"forces","threshold":0.005,"reduce_op":"norm","reduce_dims":-1
    }]),
)
for h in hooks:
    opt.register_hook(h, stage=DynamicsStage.BEFORE_COMPUTE)
t0=time.perf_counter()
result=opt.run(batch)
elapsed=time.perf_counter()-t0
energy=float(result.energy.squeeze().detach().cpu())
fmax=float(result.forces.norm(dim=-1).max().detach().cpu())

report = {
    "status":"PASS",
    "purpose":"Verify NVIDIA ALCHEMI Toolkit executes an atomistic optimization in our repo/runtime.",
    "environment":env,
    "smoke_test":{
        "model":"NVIDIA ALCHEMI LennardJonesModelWrapper",
        "optimizer":"FIRE2",
        "steps":opt.step_count,
        "final_energy_eV":energy,
        "final_fmax_eV_per_A":fmax,
        "elapsed_s":elapsed
    },
    "candidate_target":{
        "reference":"TEMED-DQ-AER",
        "counterion_candidates":["HPO4^2-","PO4^3-","P2O7^4-"],
        "control":"CO3^2-",
        "note":"No candidate binding energy is claimed by this smoke test."
    },
    "next_gate":"Build a representative TEMED-DQ local cluster and run AIMNet2/Ewald or ALCHEMI BMD on a GPU-capable runtime; compare carbonate vs phosphate hydration/binding."
}
(outdir/"alchemi-runtime-smoke-v0.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
(outdir/"alchemi-runtime-smoke-v0.md").write_text(
    "# ALCHEMI Runtime Smoke V0\n\n"
    f"**Status:** PASS\n\n"
    f"- Device: {device}\n"
    f"- CUDA available: {env['cuda_available']}\n"
    f"- ALCHEMI: {env['nvalchemi']}\n"
    f"- FIRE2 steps: {opt.step_count}\n"
    f"- Final energy: {energy:.6f} eV\n"
    f"- Final fmax: {fmax:.6f} eV/Å\n\n"
    "This proves the NVIDIA ALCHEMI code path executes. It does **not** validate the moisture-swing candidate yet.\n",
    encoding="utf-8"
)
print(json.dumps(report,indent=2))

#!/usr/bin/env python3
import json, os, random, time
from pathlib import Path

import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "03-validation"
OUT.mkdir(exist_ok=True)
RAW_PATH = OUT / "scientific-dropout-v0.json"
SUMMARY_PATH = OUT / "scientific-dropout-v0.md"

SEED, TRAIN_N, EPOCHS, BATCH = 2026, 2000, 20, 128
LR, DROP_P = 1e-3, 0.5
BASE_URL = "https://api.tokenfactory.nebius.com/v1"
PAPER = "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
SOURCE = "Journal of Machine Learning Research 15 (2014), abstract, PDF page 1"
CLAIM = "Dropout reduces overfitting and can improve held-out performance."
EVIDENCE = "This significantly reduces overfitting and gives major improvements over other regularization methods."

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)

class MLP(nn.Module):
    def __init__(self, use_dropout):
        super().__init__()
        drop = (lambda: nn.Dropout(DROP_P)) if use_dropout else (lambda: nn.Identity())
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), drop(),
            nn.Linear(512, 256), nn.ReLU(), drop(), nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.net(x)

def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total

def train_variant(use_dropout, train_loader, train_eval_loader, test_loader, device):
    seed_all(SEED)
    model = MLP(use_dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    start = time.time(); history = []
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        tr = accuracy(model, train_eval_loader, device)
        te = accuracy(model, test_loader, device)
        history.append({"epoch": epoch + 1, "train_accuracy": tr, "test_accuracy": te})
        print(("%s epoch=%d train=%.4f test=%.4f") % ("dropout" if use_dropout else "baseline", epoch+1, tr, te))
    return {
        "train_accuracy": history[-1]["train_accuracy"],
        "test_accuracy": history[-1]["test_accuracy"],
        "generalization_gap": history[-1]["train_accuracy"] - history[-1]["test_accuracy"],
        "seconds": time.time() - start,
        "history": history,
    }

def nemotron(api_key, experiment):
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    r = requests.get(BASE_URL + "/models", headers=headers, timeout=60)
    r.raise_for_status()
    ids = [x.get("id", "") for x in r.json().get("data", []) if isinstance(x, dict)]
    candidates = [m for m in ids if "nvidia" in m.lower() and "nemotron" in m.lower()
                  and not any(t in m.lower() for t in ("embed","rerank","ocr","parse"))]
    if not candidates:
        raise RuntimeError("No NVIDIA Nemotron chat model exposed by Nebius Token Factory")
    model = sorted(candidates, key=lambda x: ("nano" not in x.lower(), len(x)))[0]
    user = {
        "paper": PAPER, "claim": CLAIM, "evidence": EVIDENCE, "source": SOURCE,
        "experiment": experiment,
        "instruction": "Classify only as SUPPORTED, NOT_REPRODUCED, or INCONCLUSIVE. This is a small directional reproduction, not a full replication."
    }
    payload = {
        "model": model, "temperature": 0, "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": "Return JSON only with keys status and reason. Use only supplied evidence and observed metrics. Do not invent facts."},
            {"role": "user", "content": json.dumps(user)}
        ]
    }
    r = requests.post(BASE_URL + "/chat/completions", headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    a, b = raw.find("{"), raw.rfind("}")
    return model, json.loads(raw[a:b+1]), raw

def main():
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise SystemExit("BLOCKED: NEBIUS_API_KEY missing from runtime")
    seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = ROOT / ".cache" / "mnist"
    tr = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    te = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    idx = torch.randperm(len(tr), generator=torch.Generator().manual_seed(SEED))[:TRAIN_N].tolist()
    subset = Subset(tr, idx)
    train_loader = DataLoader(subset, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    train_eval = DataLoader(subset, batch_size=512, shuffle=False)
    test_loader = DataLoader(te, batch_size=512, shuffle=False)

    baseline = train_variant(False, train_loader, train_eval, test_loader, device)
    dropout = train_variant(True, train_loader, train_eval, test_loader, device)
    delta = dropout["test_accuracy"] - baseline["test_accuracy"]
    gap_delta = dropout["generalization_gap"] - baseline["generalization_gap"]
    experiment = {
        "seed": SEED, "device": str(device), "train_examples": TRAIN_N, "epochs": EPOCHS,
        "batch_size": BATCH, "learning_rate": LR, "dropout_p": DROP_P,
        "baseline": baseline, "dropout": dropout,
        "test_accuracy_delta": delta, "generalization_gap_delta": gap_delta
    }
    model, conclusion, raw = nemotron(api_key, experiment)
    record = {
        "paper": PAPER, "claim": CLAIM, "source": SOURCE, "evidence": EVIDENCE,
        "runtime": {"pytorch": torch.__version__, "device": str(device), "nebius_model": model},
        "experiment": experiment, "conclusion": conclusion, "nemotron_raw": raw
    }
    RAW_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    summary = f"""# Scientific Dropout V0

## Paper
{PAPER}

## Claim
{CLAIM}

## Evidence
{EVIDENCE}

Source: {SOURCE}

## Experiment
Same MNIST subset, architecture, optimizer, learning rate, epochs, batch size and seed.
Only controlled difference: dropout disabled vs enabled (p={DROP_P}).

- Seed: {SEED}
- Train examples: {TRAIN_N}
- Epochs: {EPOCHS}
- PyTorch: {torch.__version__}
- Device: {device}

## Results
- Without dropout — train: {baseline['train_accuracy']:.4f}; test: {baseline['test_accuracy']:.4f}; gap: {baseline['generalization_gap']:.4f}
- With dropout — train: {dropout['train_accuracy']:.4f}; test: {dropout['test_accuracy']:.4f}; gap: {dropout['generalization_gap']:.4f}
- Test accuracy delta: {delta:+.4f}
- Generalization-gap delta: {gap_delta:+.4f}

## NVIDIA + Nebius
NVIDIA Nemotron model {model} was called through Nebius Token Factory at runtime.

## Conclusion
{conclusion.get('status', 'UNKNOWN')}

{conclusion.get('reason', '')}

## Evidence trace
Paper → claim → exact evidence → controlled PyTorch experiment → observed metrics → NVIDIA Nemotron on Nebius → conclusion
"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print("SCIENTIFIC_VERTICAL_SLICE: PASS")
    print("NVIDIA_MODEL:", model)
    print("DEVICE:", device)
    print("BASELINE_TEST: %.4f" % baseline["test_accuracy"])
    print("DROPOUT_TEST: %.4f" % dropout["test_accuracy"])
    print("DELTA: %+.4f" % delta)
    print("STATUS:", conclusion.get("status", "UNKNOWN"))

if __name__ == "__main__":
    main()

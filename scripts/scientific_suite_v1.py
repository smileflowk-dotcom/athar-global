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
BASE_URL = "https://api.tokenfactory.nebius.com/v1"
SEED = 2026

def seed_all(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

def accuracy(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    return correct / total

def train(model, train_loader, test_loader, device, epochs, lr, optimizer="adam"):
    seed_all()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr) if optimizer == "adam" else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        te = accuracy(model, test_loader, device)
        hist.append({"epoch": epoch + 1, "test_accuracy": te})
    return {"test_accuracy": hist[-1]["test_accuracy"], "history": hist}

def nemotron(api_key, paper, claim, evidence, source, experiment):
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    r = requests.get(BASE_URL + "/models", headers=headers, timeout=60)
    r.raise_for_status()
    ids = [x.get("id", "") for x in r.json().get("data", []) if isinstance(x, dict)]
    candidates = [m for m in ids if "nvidia" in m.lower() and "nemotron" in m.lower()
                  and not any(t in m.lower() for t in ("embed", "rerank", "ocr", "parse"))]
    if not candidates:
        raise RuntimeError("No NVIDIA Nemotron chat model exposed by Nebius Token Factory")
    model = sorted(candidates, key=lambda x: ("nano" not in x.lower(), len(x)))[0]
    user = {
        "paper": paper, "claim": claim, "evidence": evidence, "source": source,
        "experiment": experiment,
        "instruction": "Classify only as SUPPORTED, NOT_REPRODUCED, or INCONCLUSIVE. Treat this as a small directional reproduction, not a full replication."
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

class PlainMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.net(x)

class BatchNormMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 10)
        )
    def forward(self, x): return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32*7*7, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x): return self.head(self.features(x))

class Cutout:
    def __init__(self, size=8): self.size = size
    def __call__(self, img):
        img = img.clone()
        h, w = img.shape[-2:]
        cy = random.randrange(h); cx = random.randrange(w)
        y1 = max(0, cy - self.size // 2); y2 = min(h, y1 + self.size)
        x1 = max(0, cx - self.size // 2); x2 = min(w, x1 + self.size)
        img[..., y1:y2, x1:x2] = 0
        return img

def fixed_subset(dataset, n):
    g = torch.Generator().manual_seed(SEED)
    return Subset(dataset, torch.randperm(len(dataset), generator=g)[:n].tolist())

def main():
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise SystemExit("BLOCKED: NEBIUS_API_KEY missing from runtime")
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = ROOT / ".cache" / "mnist"

    base_train = datasets.MNIST(cache, train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(cache, train=False, download=True, transform=transforms.ToTensor())
    train_small = fixed_subset(base_train, 2000)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    # Claim 1: Batch Normalization supports higher learning rates / easier optimization.
    train_loader = DataLoader(train_small, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    plain = train(PlainMLP(), train_loader, test_loader, device, epochs=8, lr=0.2, optimizer="sgd")
    bn = train(BatchNormMLP(), train_loader, test_loader, device, epochs=8, lr=0.2, optimizer="sgd")
    bn_exp = {
        "seed": SEED, "device": str(device), "dataset": "MNIST", "train_examples": 2000,
        "epochs": 8, "optimizer": "SGD momentum=0.9", "learning_rate": 0.2,
        "plain_test_accuracy": plain["test_accuracy"], "batchnorm_test_accuracy": bn["test_accuracy"],
        "accuracy_delta": bn["test_accuracy"] - plain["test_accuracy"],
        "controlled_difference": "BatchNorm layers only"
    }
    bn_model, bn_conclusion, bn_raw = nemotron(
        api_key,
        "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
        "Batch Normalization can make optimization at high learning rates easier.",
        "Batch Normalization allows us to use much higher learning rates and be less careful about initialization.",
        "Ioffe & Szegedy, ICML 2015, abstract",
        bn_exp
    )

    # Claim 2: Cutout can improve robustness / held-out performance.
    cut_train = datasets.MNIST(cache, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), Cutout(8)]))
    plain_subset = fixed_subset(base_train, 2000)
    cut_subset = fixed_subset(cut_train, 2000)
    plain_loader = DataLoader(plain_subset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    cut_loader = DataLoader(cut_subset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    no_cut = train(SmallCNN(), plain_loader, test_loader, device, epochs=8, lr=1e-3, optimizer="adam")
    with_cut = train(SmallCNN(), cut_loader, test_loader, device, epochs=8, lr=1e-3, optimizer="adam")
    cut_exp = {
        "seed": SEED, "device": str(device), "dataset": "MNIST", "train_examples": 2000,
        "epochs": 8, "optimizer": "Adam", "learning_rate": 0.001, "cutout_size": 8,
        "baseline_test_accuracy": no_cut["test_accuracy"], "cutout_test_accuracy": with_cut["test_accuracy"],
        "accuracy_delta": with_cut["test_accuracy"] - no_cut["test_accuracy"],
        "controlled_difference": "random 8x8 input mask during training only"
    }
    cut_model, cut_conclusion, cut_raw = nemotron(
        api_key,
        "Improved Regularization of Convolutional Neural Networks with Cutout",
        "Cutout can improve the robustness and held-out performance of convolutional neural networks.",
        "Cutout can be used to improve the robustness and overall performance of convolutional neural networks.",
        "DeVries & Taylor, arXiv:1708.04552, abstract",
        cut_exp
    )

    results = [
        {"id":"batchnorm", "experiment":bn_exp, "model":bn_model, "conclusion":bn_conclusion, "raw":bn_raw},
        {"id":"cutout", "experiment":cut_exp, "model":cut_model, "conclusion":cut_conclusion, "raw":cut_raw},
    ]
    (OUT / "scientific-suite-v1.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = [
        "# ATHAR Scientific Suite V1",
        "",
        f"Device: {device}",
        f"NVIDIA/Nebius model: {bn_model}",
        "",
        "## Batch Normalization",
        f"- Plain test accuracy: {plain['test_accuracy']:.4f}",
        f"- BatchNorm test accuracy: {bn['test_accuracy']:.4f}",
        f"- Delta: {bn_exp['accuracy_delta']:+.4f}",
        f"- Verdict: **{bn_conclusion.get('status','UNKNOWN')}**",
        f"- Reason: {bn_conclusion.get('reason','')}",
        "",
        "## Cutout",
        f"- Baseline test accuracy: {no_cut['test_accuracy']:.4f}",
        f"- Cutout test accuracy: {with_cut['test_accuracy']:.4f}",
        f"- Delta: {cut_exp['accuracy_delta']:+.4f}",
        f"- Verdict: **{cut_conclusion.get('status','UNKNOWN')}**",
        f"- Reason: {cut_conclusion.get('reason','')}",
        "",
        "## Rule",
        "Verdicts are produced from observed metrics and supplied source evidence. A negative or inconclusive result is valid and must not be rewritten as success.",
    ]
    (OUT / "scientific-suite-v1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("SCIENTIFIC_SUITE_V1: PASS")
    print("BATCHNORM:", bn_conclusion.get("status"), f"delta={bn_exp['accuracy_delta']:+.4f}")
    print("CUTOUT:", cut_conclusion.get("status"), f"delta={cut_exp['accuracy_delta']:+.4f}")

if __name__ == "__main__":
    main()

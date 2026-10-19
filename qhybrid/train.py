"""Training and evaluation loops shared by hybrid and classical models."""
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn


def train_model(model, Ztr, ytr, epochs=15, lr=0.01, batch_size=64, seed=0, val=None):
    """Adam + cross-entropy. If val=(Z, y) is given, macro-F1 on it is recorded after each epoch."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Z, y = torch.as_tensor(Ztr), torch.as_tensor(ytr)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Z))
        total = 0.0
        for i in range(0, len(Z), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Z[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        rec = {"loss": total / len(Z)}
        if val is not None:
            rec["val_f1"] = evaluate(model, *val)["f1"]
        history.append(rec)
    return history


@torch.no_grad()
def evaluate(model, Zte, yte, batch_size=256):
    model.eval()
    Z = torch.as_tensor(Zte)
    t0 = time.perf_counter()
    pred = torch.cat([model(Z[i:i + batch_size]).argmax(-1) for i in range(0, len(Z), batch_size)])
    dt = time.perf_counter() - t0
    pred = pred.numpy()
    return {"acc": accuracy_score(yte, pred), "f1": f1_score(yte, pred, average="macro"),
            "ms_per_img": 1000 * dt / len(Z)}


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

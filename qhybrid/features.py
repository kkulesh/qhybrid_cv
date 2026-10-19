"""Feature extraction with a frozen YOLO11-cls backbone (penultimate 1280-d vector)."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO



def list_images(root, per_class=None, seed=0):
    """Return (paths, labels, class_names) for an ImageFolder-style directory."""
    root = Path(root)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = np.random.default_rng(seed)
    paths, labels = [], []
    for ci, c in enumerate(classes):
        files = sorted((root / c).glob("*"))
        if per_class is not None and len(files) > per_class:
            files = [files[i] for i in rng.choice(len(files), per_class, replace=False)]
        paths += files
        labels += [ci] * len(files)
    return paths, np.array(labels), classes


class YoloFeatureExtractor:
    def __init__(self, weights="yolo11n-cls.pt", imgsz=224, device="cpu"):
        self.model = YOLO(weights).model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.imgsz, self.device = imgsz, device
        self._feat = None
        head = self.model.model[-1]
        head.linear.register_forward_hook(lambda m, inp, out: setattr(self, "_feat", inp[0].detach()))

    def _load(self, path):
        return self.to_tensor(Image.open(path))

    def to_tensor(self, img):
        """PIL image -> 3 x imgsz x imgsz float tensor in [0, 1]."""
        img = img.convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
        return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)

    @torch.no_grad()
    def extract_tensor(self, x):
        """Features for a batch tensor already prepared by to_tensor."""
        self.model(x.to(self.device))
        return self._feat.cpu()

    @torch.no_grad()
    def extract(self, paths, batch_size=64):
        feats = []
        for i in range(0, len(paths), batch_size):
            x = torch.stack([self._load(p) for p in paths[i:i + batch_size]]).to(self.device)
            self.model(x)
            feats.append(self._feat.cpu().numpy())
        return np.concatenate(feats).astype(np.float32)

from __future__ import annotations

import csv

import torch
from torch.utils.data import Dataset

from src.localizer.geometry import REF_DS_PX
from src.localizer.inference import load_standardized


class ManifestDataset(Dataset):

    def __init__(self, manifest_path: str):
        with open(manifest_path, newline="") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        ref = load_standardized(row["reference_path"], REF_DS_PX).squeeze(0)
        search = load_standardized(row["search_path"]).squeeze(0)
        return {
            "reference_img": ref,
            "search_img": search,
            "gt_x": torch.tensor(float(row["gt_x"]), dtype=torch.float32),
            "gt_y": torch.tensor(float(row["gt_y"]), dtype=torch.float32),
        }

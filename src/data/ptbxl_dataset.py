import ast
import os

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

LABEL_MAP = {
    "NORM": 0,
    "MI":   1,
    "STTC": 2,
    "CD":   3,
    "HYP":  4,
}

DATA_DIR_DEFAULT = "./data/ptbxl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"


class PTBXLDataset(Dataset):

    def __init__(self, data_dir: str = DATA_DIR_DEFAULT, sampling_rate: int = 100, split: str = "train"):
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate

        # Metadata
        df = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        # Tanı kodu → süper sınıf tablosu
        agg_df = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1.0]

        # Etiket ata
        df["label"] = df["scp_codes"].apply(lambda x: self._get_label(x, agg_df))

        # Etiketsiz kayıtları çıkar
        df = df[df["label"] != -1]

        # Resmi train/test split (fold 10 → test)
        if split == "train":
            self.df = df[df["strat_fold"] != 10].reset_index()
        else:
            self.df = df[df["strat_fold"] == 10].reset_index()

        print(f"[PTBXLDataset] {split}: {len(self.df)} kayıt | "
              f"sınıf dağılımı: { {k: int((self.df.label == v).sum()) for k, v in LABEL_MAP.items()} }")

    # ------------------------------------------------------------------
    def _get_label(self, scp_codes: dict, agg_df: pd.DataFrame) -> int:
        for code, prob in sorted(scp_codes.items(), key=lambda x: -x[1]):
            if code in agg_df.index:
                superclass = agg_df.loc[code, "diagnostic_class"]
                if superclass in LABEL_MAP:
                    return LABEL_MAP[superclass]
        return -1

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        if self.sampling_rate == 100:
            path = os.path.join(self.data_dir, row["filename_lr"])
        else:
            path = os.path.join(self.data_dir, row["filename_hr"])

        signal, _ = wfdb.rdsamp(path)           # (1000, 12) veya (5000, 12)

        # Z-score normalize
        mean = signal.mean(axis=0, keepdims=True)
        std  = signal.std(axis=0, keepdims=True) + 1e-8
        signal = (signal - mean) / std

        x = torch.tensor(signal, dtype=torch.float32)  # (1000, 12)
        y = torch.tensor(row["label"], dtype=torch.long)
        return x, y

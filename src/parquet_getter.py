import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq

from pathlib import Path

class SpectrogramDataset(IterableDataset):
    def __init__(self, n_channels=2, n_mels=64):
        src_dir = Path(__file__).resolve().parent
        parq_dir = src_dir / "parq_data"
        parq_file = parq_dir / "dataset.parquet"
        self.parquet_path = parq_file
        self.n_channels = n_channels
        self.n_mels = n_mels

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)

        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            batch = table.to_pydict()

            specs = batch["spec"]
            labels = batch["label"]

            for spec_flat, label in zip(specs, labels):
                spec = torch.tensor(spec_flat, dtype=torch.float32)
                spec = spec.view(self.n_channels, self.n_mels, -1)
                yield spec, torch.tensor(label, dtype=torch.long) # return the tensor and dtype
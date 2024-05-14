import torch
import numpy as np
import pandas as pd


def dataset_loader(root_path: str, dataset_name: str, file_idx: int):
    if dataset_name == "BGP":
        return torch.load(f'{root_path}/{dataset_name}/cached_dataset_{file_idx}.pt')
    else:
        return torch.load(f'{root_path}/{dataset_name}/cached_dataset.pt')

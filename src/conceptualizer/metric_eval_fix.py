import random

import torch
from torch.utils.data import DataLoader, Dataset


def get_loader(base_datasets, batch_size):
    arch_flag = "concepts"

    class DictDataset(Dataset):
        def __init__(self, data_dict):
            self.data_dict = data_dict
            self.keys = list(data_dict.keys())
            self.lengths = [len(data) for data in data_dict.values()]
            self.total_length = sum(self.lengths)

        def __len__(self):
            if arch_flag == "vanilla-obs":
                return len(self.data_dict["obs"])
            return self.total_length

        def __getitem__(self, item):
            idx, key = item
            dataset = self.data_dict[key]
            return dataset[idx], key

    class DictBatchSampler:
        def __init__(self, data_dict, batch_size):
            self.data_dict = data_dict
            self.keys = list(data_dict.keys())
            self.batch_size = batch_size
            self.total_samples = sum(len(dataset) for dataset in data_dict.values())
            if arch_flag == "vanilla-obs":
                self.total_samples = len(self.data_dict["obs"])

        def __iter__(self):
            samples_yielded = 0
            while samples_yielded < self.total_samples:
                key = random.choice(self.keys)
                if arch_flag == "vanilla-obs":
                    key = "obs"
                dataset = self.data_dict[key]
                remaining = min(self.batch_size, self.total_samples - samples_yielded)
                indices = torch.randperm(len(dataset))[:remaining]
                yield [(idx.item(), key) for idx in indices]
                samples_yielded += len(indices)

        def __len__(self):
            return (self.total_samples + self.batch_size - 1) // self.batch_size

    def dict_collate_fn(batch):
        data = torch.stack(
            [item[0][0] if isinstance(item[0], tuple) else item[0] for item in batch]
        )
        key = batch[0][1]  # All keys in the batch are the same
        return data, key

    dataset = DictDataset(base_datasets)
    batch_sampler = DictBatchSampler(base_datasets, batch_size=batch_size)
    data_loader = DataLoader(
        dataset, batch_sampler=batch_sampler, collate_fn=dict_collate_fn
    )
    return data_loader

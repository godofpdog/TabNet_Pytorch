""" Implementations of trainig operations """

import time
import torch 
from torch.utils.data import DataLoader


def train_epoch(
    model, datasets, epoch, batch_szie, criterion, optimizer,
    metrics=None, device='cpu', model_type='inference_model', 
    is_shuffle=True, sampler=None, pin_memory=False, num_workers=1
):
    """

    """
    torch.cuda.empty_cache()
    model.train()

    total_loss = 0

    data_loader = DataLoader(
        datasets, batch_size=batch_szie, shuffle=is_shuffle, sampler=sampler, 
        pin_memory=pin_memory, num_workers=num_workers
    )

    for i, data in enumerate(data_loader):
        # NOTE must consider multi-task 

        start = time.time()



import torch


def get_batch(split, batch_size, block_size, device):
    ix = torch.randint(len(split) - batch_size, (batch_size,))
    # ix = (B, 1)
    x = torch.stack([split[i:i+block_size] for i in ix])
    # (B, block_size)
    y = torch.stack([split[i+1: i+1+block_size] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y
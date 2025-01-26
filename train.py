from utils.vanilla_tokenizers import extract_encoder_decoder, test_encoder_decoder
from utils.split_data import get_batch
from model.bigram_model import BigramModel
from tqdm import tqdm
from config import get_default_config, get_weights_file_path, get_latest_weights_file_path, ModelConfig

#Distributed training
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import torch
import argparse
import os
import torch


@torch.no_grad()  # doesn't calculate gradients. We will not call .backward()
def estimate_loss(model, train_data, val_data, eval_iters, device, batch_size, max_length):
    out = {}
    model.eval()  # layers like batchnorm and dropout won't be applied
    for split in ['train', 'val']:
        data = train_data if split == "train" else val_data
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(data, batch_size, max_length, device)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train_distributed(config):
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])

    init_process_group(backend='ncc1')
    torch.cuda.set_device(local_rank) # set the device to the local rank within the server (global rank)

    if global_rank == 0:
        # start W&B, etc.
        pass

    data_loader = DataLoader(train_dataset, shuffle=False, sampler = DistributedSampler(train_dataset, shuffle=True))
    model = Model()

    if os.path.exists('latest_checkpoint.pth'):
        model.load_state_dict(torch.load('latest_checkpoint.pth'))
    
    model = DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        for data, labels in data_loader:
            loss = loss_fn(model(data), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if global_rank == 0:
                # collect_statistics()
                # W&B etc.
                pass
        
        if global_rank == 0:
            torch.save(model.state_dict(), 'latest_checkpoint.pth')

    destroy_process_group()



def train(config):

    assert torch.cuda.is_available(), "Training on CPU is not supported"

    filename = "data/input.txt"

    encoder, decoder, text, vocab_size = extract_encoder_decoder(filename)
    data = torch.tensor(encoder(text), dtype=torch.long)

    split = int(0.9*len(data))
    train = data[:split]
    val = data[split:]

    max_length = config.seq_len  # x = 0 y = 1, x = 0, 1 y = 2, x = 0,1,2 y =3 ... so on till x is len 8
    batch_size = config.batch_size
    lr = config.lr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = config.num_epochs
    eval_interval = 500
    embed_size = config.d_model
    attention_heads = 6
    dropout = 0.2

    m = BigramModel(vocab_size, embed_size, max_length,
                          attention_heads, dropout, device)
    m = m.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    for e in tqdm(range(epochs)):

        # every eval_interval, eval how we are doing on the loss
        if e % eval_interval == 0:
            losses = estimate_loss(
                m, train, val, eval_interval, device, batch_size, max_length)
            print(
                f"step {e}: train loss: {losses['train']:.4f}, val loss {losses['val']:.4f}")

        train_x_batch, train_y_batch = get_batch(
            train, batch_size, max_length, device)
        logits, loss = m(train_x_batch, train_y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss)

    # val_x_batch, val_y_batch = get_batch(val, batch_size, max_length)

    logits, loss = m(train_x_batch, train_y_batch)

    print(logits.shape)
    # start index with (B,1)
    start_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

    # Generate 5 chars and grab first row
    output = m.generate(start_idx, 250)[0].tolist()
    print("".join(decoder(output)))


if __name__ == "__main__":

    config = get_default_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--num_epochs', type=int, default=config.num_epochs)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--seq_len', type=int, default=config.seq_len)
    parser.add_argument('--d_model', type=int, default=config.d_model)
    parser.add_argument('--lang_src', type=str, default=config.lang_src)
    parser.add_argument('--lang_tgt', type=str, default=config.lang_tgt)
    parser.add_argument('--model_folder', type=str, default=config.model_folder)
    parser.add_argument('--model_basename', type=str, default=config.model_basename)
    parser.add_argument('--preload', type=str, default=config.preload)
    parser.add_argument('--tokenizer_file', type=str, default=config.tokenizer_file)
    parser.add_argument('--wandb_group', type=str, default="exp1")
    args = parser.parse_args()


    # config.__dict__.update(vars(args))
    
    # # global and local ranks to config
    # config.local_rank = int(os.environ['LOCAL_RANK'])
    # config.global_rank = int(os.environ['GLOBAL_RANK'])

    # assert config.local_rank != -1, "LOCAL_RANK environment variable not set"
    # assert config.global_rank != -1, "RANK environment variable not set"

    # # print a single configuration, once per server (local rank being 0)
    # if config.local_rank == 0:
    #     print("Configuration:")
    #     for key, value in config.__dict__.items():
    #         print(f"{key:>20}: {value}")
    

    # # setup distributed training
    # init_process_group(backend='ncc1')
    # torch.cuda.set_device(config.local_rank)

    train(config)


    #clean up distributed training
    destroy_process_group()


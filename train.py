from utils.vanilla_tokenizers import extract_encoder_decoder, test_encoder_decoder
from utils.split_data import get_batch
from model.bigram_model import Transformer_Model
from tqdm import tqdm
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


def train():
    filename = "data/input.txt"

    encoder, decoder, text, vocab_size = extract_encoder_decoder(filename)
    data = torch.tensor(encoder(text), dtype=torch.long)

    split = int(0.9*len(data))
    train = data[:split]
    val = data[split:]

    max_length = 256  # x = 0 y = 1, x = 0, 1 y = 2, x = 0,1,2 y =3 ... so on till x is len 8
    batch_size = 64
    lr = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5000
    eval_interval = 500
    embed_size = 384
    attention_heads = 6
    dropout = 0.2

    m = Transformer_Model(vocab_size, embed_size, max_length,
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


train()

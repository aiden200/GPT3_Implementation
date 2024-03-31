from utils.vanilla_tokenizers import extract_encoder_decoder, test_encoder_decoder
from utils.split_data import get_batch
from model.bigram_model import BigramModel
from tqdm import tqdm
import torch


@torch.no_grad() # doesn't calculate gradients. We will not call .backward()
def estimate_loss(model, train_data, val_data, eval_iters, device, batch_size, block_size):
    out = {}
    model.eval() # layers like batchnorm and dropout won't be applied
    for split in ['train', 'val']:
        data = train_data if split == "train" else val_data
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            logits, loss = model(X,Y)
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




    block_size = 30 # x = 0 y = 1, x = 0, 1 y = 2, x = 0,1,2 y =3 ... so on till x is len 8
    batch_size = 32
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10000
    eval_interval = 500
    embed_size = 128
    attention_hidden_dim = 128
    attention_heads = 6
    ffn_hidden_dim = 128
    ffn_layers = 2

    m = BigramModel(vocab_size, embed_size, block_size, attention_hidden_dim, attention_heads, ffn_hidden_dim, ffn_layers, device)
    m = m.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    for e in tqdm(range(epochs)):

        # every eval_interval, eval how we are doing on the loss
        if e % eval_interval == 0:
            losses = estimate_loss(m, train, val, eval_interval, device, batch_size, block_size)
            print(f"step {e}: train loss: {losses['train']:.4f}, val loss {losses['val']:.4f}")


        train_x_batch, train_y_batch = get_batch(train, batch_size, block_size, device)
        logits, loss = m(train_x_batch, train_y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print(loss)

    # val_x_batch, val_y_batch = get_batch(val, batch_size, block_size)




    logits, loss = m(train_x_batch, train_y_batch)

    print(logits.shape)
    start_idx = torch.zeros((batch_size, 1), dtype=torch.long, device=device) # start index with (B,1)

    output = m.generate(start_idx, 20)[0].tolist() # Generate 5 chars and grab first row
    print("".join(decoder(output)))


train()






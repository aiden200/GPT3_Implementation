from utils.vanilla_tokenizers import extract_encoder_decoder, test_encoder_decoder
import torch





def train():
    filename = "data/input.txt"

    encoder, decoder, text = extract_encoder_decoder(filename)
    data = torch.tensor(encoder(text), dtype=torch.long)

    split = int(0.9*len(data))
    train = data[:split]
    val = data[split:]

    block_size = 8 # x = 0 y = 1, x = 0, 1 y = 2, x = 0,1,2 y =3 ... so on till x is len 8
    batch_size = 4

    



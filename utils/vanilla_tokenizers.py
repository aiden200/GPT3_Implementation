import torch
import torch.nn as nn
from torch.nn import functional as F


def extract_encoder_decoder(filename: str) -> tuple:
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    print("length of dataset in characters: ", len(text))
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(''.join(chars))
    print(f"Vocab Size: {vocab_size}")

    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(s): return [char_to_int[c] for c in s]
    def decode(l): return [int_to_char[i] for i in l]

    return (encode, decode, text, vocab_size)


def test_encoder_decoder(filename):
    test_str = "This is a test"
    encoder, decoder = extract_encoder_decoder(filename)

    print(f"Encoded: {encoder(test_str)}")
    print(f"Decoded: {decoder(encoder(test_str))}")
    assert test_str == "".join(decoder(encoder(test_str)))

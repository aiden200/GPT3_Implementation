import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer_blocks import PositionalEncoding, Transformer_block


class Bert(nn.Module):
    def __init__(self, num_layers, block_size, vocab_size, embed_size, attention_heads, attn_dropout, layer_dropout, device):
        self.embedding_table = nn.Embedding(vocab_size+1, embed_size) #++1 for mask
        self.pe = PositionalEncoding(embed_size)
        self.layers = nn.ModuleList(
            [Transformer_block(embed_size, block_size,attention_heads, attn_dropout, device) for _ in range(num_layers)]
        )
        self.lin_out = nn.Linear(embed_size, vocab_size)
    
    def forward(self, idx, targets = None):
        x = self.embedding_table(idx)
        x = self.pe(x)
        for t in self.layers:
            x = t(x)
        x = self.lin_out(x)
        out = nn.Sigmoid(x)
        return out

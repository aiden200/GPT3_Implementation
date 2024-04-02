import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, drop_rate=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=drop_rate)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    

class Transformer_block(nn.Module):
    def __init__(self, embed_size, block_size, attention_heads, dropout, device):
        super().__init__()

        self.block_size = block_size  # block size is maximum length
        self.device = device

        # for each char, add positional encoding to embed
        self.pe = nn.Embedding(block_size, embed_size)

        self.masked_mh_self_attention = Multi_head_attention(
            embed_size//attention_heads, embed_size, block_size, attention_heads, dropout)
        self.attn_norm = nn.LayerNorm(embed_size)
        self.ffn = FeedForward(embed_size)
        self.ffn_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Masked multi-head self attention
        resid_x = x
        x = self.masked_mh_self_attention(x)

        # Add & Norm
        x = x + resid_x
        x = self.attn_norm(x)

        # Feedforward Network
        resid_x = x
        x = self.ffn(x)  # (B, T, H) where h is hidden dim for

        # add & norm
        x = x + resid_x
        out = self.ffn_norm(x)

        return out




class Multi_head_attention(nn.Module):
    def __init__(self, hidden_dim, token_embed_dim, max_length, attention_heads, dropout, masked=True) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Attention_Head(hidden_dim, token_embed_dim, max_length,
                            masked, dropout) for _ in range(attention_heads)]
        )

        # projection back into original dim for residual connections
        self.proj = nn.Linear(hidden_dim * attention_heads, token_embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # heads * (B, T, H) We want to concat in the H dimension.
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.proj(x)
        out = self.dropout(x)
        return out


class FeedForward(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden_dim = 4 * input_dim  # 4x is what they did in attention is all you need paper

        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim), nn.ReLU()])
        # for proj. back to original dim for resid
        self.layers.append(nn.Linear(hidden_dim, input_dim))

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class Attention_Head(nn.Module):
    # one head of self attention
    def __init__(self, hidden_dim, token_embed_dim, block_size, masked, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.k_w = nn.Linear(token_embed_dim, self.hidden_dim,
                             bias=False)  # (B, T, hidden_dim)
        # (B, T, hidden_dim) token embeddings -> number of hidden_dim
        self.q_w = nn.Linear(token_embed_dim, self.hidden_dim, bias=False)
        self.q_v = nn.Linear(token_embed_dim, self.hidden_dim,
                             bias=False)  # (B, T, hidden_dim)
        # self.tril = torch.tril(torch.ones(block_size, block_size)) # lower triangular matrix only used for decoder!
        self.decoder = False
        if masked:
            self.decoder = True
            self.register_buffer('tril', torch.tril(
                torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, E), e is token_embed dim
        T = x.shape[1]
        key = self.k_w(x)
        query = self.q_w(x)

        # (B, T, H) @ (B, H, T) -> (B, T, T)
        qk = query @ key.transpose(-2, -1)
        # qk is the relations between every token's query(what they're looking for) & every token's key(what they contain)

        if self.decoder:
            # qk = qk.masked_fill(self.tril==0, float('-inf')) # fill in all the future tokens as 0
            qk = qk.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        qk = qk * 1/math.sqrt(self.hidden_dim)  # Scale by 1/sqrt(d_k)
        # equiv to dim =1 in 2d, each row sums up to 1
        probs = F.softmax(qk, dim=-1)

        probs = self.dropout(probs)  # Randomly 0 out some attentions

        v = self.q_v(x)

        out = probs @ v  # (B, T, T) @ (B, T, hidden_dim) -> (B, T, hidden_dim)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_Block:
    # one head of self attention
    def __init__(self, hidden_dim, token_embed_dim, block_size):
        self.hidden_dim = hidden_dim
        self.k_w = nn.Linear(token_embed_dim, self.hidden_dim, bias=False) # (B, T, hidden_dim)
        self.q_w = nn.Linear(token_embed_dim, self.hidden_dim, bias=False) # (B, T, hidden_dim) token embeddings -> number of hidden_dim
        self.q_v = nn.Linear(token_embed_dim, self.hidden_dim, bias=False) # (B, T, hidden_dim)
        # self.tril = torch.tril(torch.ones(block_size, block_size)) # lower triangular matrix only used for decoder!
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        # x: (B, T, E), e is token_embed dim
        T = x.shape[1] 
        key = self.k_w(x)
        query = self.q_w(x)
        
        qk = query @ key.transpose(-2, -1) # (B, T, H) @ (B, H, T) -> (B, T, T)
        #qk is the relations between every token's query(what they're looking for) & every token's key(what they contain)
        
        # qk = qk.masked_fill(self.tril==0, float('-inf')) # fill in all the future tokens as 0
        qk = qk.mased_fill(self.tril[:T, :T] == 0, float('inf'))
        qk = qk * 1/torch.sqrt(self.hidden_dim) # Scale by 1/sqrt(d_k)
        probs = F.softmax(qk, dim=-1) # equiv to dim =1 in 2d, each row sums up to 1
        
        v = self.q_v(x)
        
        out = probs @ v # (B, T, T) @ (B, T, hidden_dim) -> (B, T, hidden_dim)
        return out
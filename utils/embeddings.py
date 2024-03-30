import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size, embed_size, block_size, device):
        super().__init__()
        self.device = device
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.pe = nn.Embedding(block_size, embed_size) # for each char, add positional encoding to embed
        self.lm_head = nn.Linear(embed_size, vocab_size)

        
    def forward(self, idx, targets=None):
        tok_embeddings = self.token_embedding_table(idx) # (B, T, E) T here is the block_size
        
        pe = self.pe(torch.arrange(idx.shape[1], device=self.device)) # (T, E), B gets broadcasted from pe + x
        
        tok_embeddings = tok_embeddings + pe
        
        logits = self.lm_head(tok_embeddings) # (B, T, V)
        

        # C is vocab size
        B, T, C = logits.shape

        if targets == None:
            loss = None
        else:
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

            logits = logits.view(B, T, C)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            #predictions
            logits, loss =self(idx)
            
            last_char_logits = logits[:, -1, :] # last character

            probs = F.softmax(last_char_logits, dim=1) # Probabilities across all chars

            idx_next = torch.multinomial(probs, num_samples=1) #(B,1) Not the highest prob but sample from probabilities

            idx = torch.cat((idx, idx_next), dim=1) # We add to the time (B, T +1) 
        
        return idx # We return (B, T + max_new_tokens)  


 

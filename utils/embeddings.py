import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C) T here is the block_size

        # C is all of the given chars
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


 

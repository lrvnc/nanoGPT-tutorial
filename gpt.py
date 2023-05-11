"""
    Bigram model with self attention.
    A bla bla model.
    Pre-training stage of LLMs, such as chatGPT.
    The model is trained to predict the next token in a sequence given the previous tokens.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}", end='\n\n')
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 3 
dropout = 0.1
# ------------

# Best hyperparameters (you'll need a GPU)
# batch_size = 64 
# block_size = 256 
# max_iters = 5000
# eval_interval = 300
# learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}", end='\n\n')
# eval_iters = 200
# n_embd = 384
# n_head = 6 # head_size = 64
# n_layer = 6
# dropout = 0.1
# ------------

torch.manual_seed(1337)

# Reading our data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("Length of text: {} characters".format(len(text)), end='\n\n')

# Let's create our vocabulary, based on our data
chars = sorted(list(set(text))) # set() removes duplicates
vocab_size = len(chars) # We will call it C, from Channel

# Tokenize functions
char2idx = {u:i for i, u in enumerate(chars)}
idx2char = {i:u for i, u in enumerate(chars)}

# Here we create two functions to encode and decode our text
tokenize = lambda text: [char2idx[c] for c in text] # encode text to numbers
detokenize = lambda indexes: ''.join([idx2char[c] for c in indexes]) # decode indexes to text


# Let's tokenize our entire text and split
data = torch.tensor(tokenize(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# Data loader
def get_batch(split):
    # Generate a smal batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,)) # random start indices for the examples
    x = torch.stack([data[i:i+block_size] for i in ix]) # (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # targets (the same as inputs, but shifted one character ahead)
    return x.to(device), y.to(device)


# Evaluation function
@torch.no_grad()
def estimate_loss():
    # Evaluate the model on the validation set
    # Performs batch average over the validation set
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Self-Attention Head
class Head(nn.Module):
    """
        A single attention head.
    """
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # in buffer, not in parameters
        self.dropout = nn.Dropout(dropout)

    def forward(self, embd_x):
        B, T, C = embd_x.shape # T can vary
        k = self.key(embd_x) # (B, T, head_size)
        q = self.query(embd_x) # (B, T, head_size)
        # Compute affinities
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,H) @ (B,H,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, value=float('-inf')) # (T,T) -> Masking wei (affinity)
        wei = F.softmax(wei, dim=-1) # (B,T,T) -> Log softmax
        wei = self.dropout(wei) # Regularization to prevent affinity overfitting
        # Weighted aggregation
        v = self.value(embd_x) # (B,T,H)
        out = wei @ v # (B,T,T) @ (B,T,H) -> (B,T,H)
        return out
    
# Multi-Head Self-Attention
class MultiHeadAttention(nn.Module):
    """Multiple Self-Attention Heads in parallel."""

    def __init__(self, n_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads*head_size, n_embd) # Projection layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,H) -> (B,T,n_heads*head_size)
        # Projecting back to the original dimensionality
        out = self.proj(out) # (B,T,n_heads*head_size) -> (B,T,n_embd)
        out = self.dropout(out) # Regularization to prevent overfitting
        return out
    
# Simple Multi-Layer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.Linear(4*n_embd, n_embd), # Projection to the original pathway
            nn.Dropout(dropout) # Regularization to prevent overfitting
        )

    def forward(self, x):
        return self.net(x)
    
# A Transformer Block
class Block(nn.Module):
    """Comminucation followed by computation.
    
    We use residual connections to improve the model's trainability.
    A residual connection, is a technique used in deep neural networks to improve the
    flow of gradients and enable the training of deeper models.
    It involves adding the skip_connection directly to the input, allowing the gradients
    to bypass certain layers and propagate more easily through the network.
    
    """
    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size) # communication
        self.ffwd = MLP(n_embd) # computation
        self.ln1 = nn.LayerNorm(n_embd) # Layer normalization (pre-normalization)
        self.ln2 = nn.LayerNorm(n_embd) # Layer normalization (post-normalization)

    def forward(self, x_embd):
        out = x_embd + self.sa_heads(self.ln1(x_embd)) # residual connection
        out = out + self.ffwd(self.ln2(out)) # residual connection
        return out


# The Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # Encoding with more interactions between tokens
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional encoding
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # (C, n_embd) => (C, C) to get logits for each token

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers

        B, T = idx.shape
        # To encode the position of the tokens
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)

        # The logits are the "scores" for the next character in the sequence
        tok_emb = self.token_embedding_table(idx) # (B, T, C)

        # We sum the two embeddings
        x_emb = tok_emb + pos_emb # (B, T, C) -> broadcast accross the batch dimension

        # We pass the embeddings through the transformer blocks:
        #  - Communication (Multi-Head Self-Attention)
        #  - Computation (MLP)
        out = self.ln(self.blocks(x_emb)) # (B, T, C)

        # Compute the logits
        logits = self.lm_head(out) # (B, T, vocab_size)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # (B*T, C) => Flatten to pass to nn.CrossEntropyLoss
            targets = targets.view(B*T) # (B*T) => Flatten to pass to nn.CrossEntropyLoss
            loss = F.cross_entropy(logits, targets)
            return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # This is the function we will use to generate new text given a context

        # idx is a (B, T) tensor of integers in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens (due to the position embedding)
            idx_cond = idx[:,-block_size:] # (B, T) => (B, min(T, block_size))
            logits, loss = self(idx_cond) # Get the predictions
            logits = logits[:,-1,:] # Focus only on the last time step => (B, C)
            probs = F.softmax(logits, dim=-1) # Turn logits into probabilities
            new_tokens = torch.multinomial(probs, num_samples=1) # Sample from the distribution => (B, 1)
            idx = torch.cat([idx, new_tokens], dim=1) # Append the new tokens to the current context => (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)


# Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(detokenize(m.generate(context, max_new_tokens=1000)[0].tolist()))
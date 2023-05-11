import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}", end='\n\n')
eval_iters = 200
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


# The Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Each token directly read off the lofigs for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        # The logits are the "scores" for the next character in the sequence
        logits = self.token_embedding_table(idx) # (B, T, C)
        
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
            logits, loss = self(idx) # Get the predictions
            logits = logits[:,-1,:] # Focus only on the last time step => (B, C)
            probs = F.softmax(logits, dim=-1) # Turn logits into probabilities
            new_tokens = torch.multinomial(probs, num_samples=1) # Sample from the distribution => (B, 1)
            idx = torch.cat([idx, new_tokens], dim=1) # Append the new tokens to the current context => (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)


# Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

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
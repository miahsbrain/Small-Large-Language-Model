import torch
from torch import nn
from torch.nn import functional as F

# Hyper params
batch_size = 32
block_size = 256
max_iters = 5000
eval_internal = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_mbed = 384
n_head = 6
n_layer = 6
dropout = 0.2


torch.manual_seed(42)
torch.cuda.manual_seed(42)
# Open file and setp encode and decode function
with open('input.txt', 'r') as f:
    text = f.read()
print(f'The length of text is {len(text)}')

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {k:v for v,k in enumerate(chars)}
itos = {k:v for k,v in enumerate(chars)}
encode = lambda word: [stoi[x] for x in word]
decode = lambda word: ''.join([itos[x] for x in word])

# Train and test split
data = torch.tensor(encode(text), dtype=torch.int64)
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    # Generate small batch of data for X and y inputs
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, [batch_size])
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

torch.manual_seed(42)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_mbed, head_size, bias=False)
        self.key = nn.Linear(n_mbed, head_size, bias=False)
        self.value = nn.Linear(n_mbed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)
        # Compute the attention scores
        wei = k @ q.transpose(-1, -2) * C**-0.5 # returns (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = wei.softmax(dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_mbed, n_mbed)
        self.dropout = nn.Dropout(dropout)

        # assert(head_size // num_heads == 0), 'Head size must be divisible by number of heads'

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_mbed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mbed, 4 * n_mbed),
            nn.ReLU(),
            nn.Linear(4 * n_mbed, n_mbed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    '''Transformer block, communication followed by computation'''
    def __init__(self, n_mbed, n_head):
        super().__init__()
        head_size = n_mbed // n_head
        self.sa = MultiHeadAttention(n_head, head_size) # 4 heads of 8 dimensional self attention.. n_mbed=32
        self.ff = FeedForward(n_mbed)
        self.ln1 = nn.LayerNorm(n_mbed)
        self.ln2 = nn.LayerNorm(n_mbed)

        assert(n_mbed % n_head == 0), 'n_mbed must be divisible by n_head'

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.sa(self.ln1(x)) + x
        x = self.ff(self.ln2(x)) + x
        return x

# class LayerNorm(nn.Module):
#     def __init__(self, dim, eps = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.gamma = torch.zeros(dim)
#         self.beta = torch.ones(dim)

#     def __call__(self, x):
#         xmean = x.mean(1, keepdim=True) # Layer mean
#         xvar = x.var(1, keepdim=True) # Layer var
#         xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # Normalize to unit variance
#         self.out = self.gamma * xhat + self.beta
#         return self.out
    
#     def parameters(self):
#         return [self.gamma, self.beta]

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_mbed)
        self.position_embedding_table = nn.Embedding(block_size, n_mbed)
        self.blocks = nn.Sequential(*[Block(n_head=n_head, n_mbed=n_mbed) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_mbed)
        self.lm_head = nn.Linear(in_features=n_mbed, out_features=vocab_size)

    def forward(self, idx: torch.tensor, targets: torch.tensor = None) -> torch.tensor:
        B, T = idx.shape
        # print(x.shape)
        tok_embed = self.token_embedding_table(idx) # (B, T, C=32)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) positional embedding'
        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.lnf(x)
        logits = self.lm_head(x) # (B, T, C=vocab_size)

        if targets is None:
            loss = None
        else:
            # print(f'Logits shape before sizing down: {logits.shape}')
            # print(f'logits permute {logits.permute(0,2,1).shape}')
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            # print(f'Logits shape after sizing down: {logits.shape}')
            # print(f'Targets before view: {targets}')
            targets = targets.view(B*T)
            # print(f'Targets after view: {targets}')
            loss = F.cross_entropy(logits, targets)
            # print(loss)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # print(idx)
            logits, loss = self(idx_cond)
            # focus only on the last timestep
            logits = logits[:, -1, :] # becomes (B, C)
            # get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # print(probs)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # print(f'\nidx_next: {probs.argmax(dim=1)}')
            # print(f'\nidx_next: {idx_next}')
            # apply sampled index to the running index
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx
        
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.Adam(params=m.parameters(), lr=1e-3)

for iter in range(max_iters):
    if ((iter + 1) % eval_internal == 0) or ((iter + 1) == max_iters):
        losses = estimate_loss()
        print(f'Epoch: {iter + 1} | Train loss: {losses["train"]:.4f} | Test loss: {losses["test"]:.4f}')
    
    # Set model to train
    m.train()

    # Get smaple batch batch
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    
    # Evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros(size=[1,1], dtype=torch.int64, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
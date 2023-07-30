import torch
import torch.nn as nn
from torch.nn import functional as F

#parameters
#independent sequences processed in parallel
batch_size = 64
#maximum context length for predictions
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4  # self-attention can't have high learning rates
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)


# read it in to inspect it
with open('harrypotter2.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#tokenise text
#convert raw text as a string to some sequence of integers
#translate characters into integers
#iterate through characters and create lookup table from char to ints
stoi = {ch: i for i, ch in enumerate(chars)}
#vice versa
itos = {i:ch for i, ch in enumerate(chars)}
#encode: take string and output list of ints
encode = lambda s: [stoi[c] for c in s]
#decode:: ake list of ints and output string
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
#split data into training and test set
#help w overfitting
#first 90% will be training and rest testing
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#data loading 
def get_batch(split):
  #generate small batch of data of inputs x and targets y
  #get data array
  if split == 'train':
    data = train_data
  else:
    data = val_data
    #four numbers randomly generated - random offsets into the training set
  ix = torch.randint(len(data) - block_size, (batch_size,))
  #first block size characters - stack into rows 4x8 tensor
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  #move data to device
  x, y = x.to(device), y.to(device)
  return x, y

#no back propogation
@torch.no_grad()
#average out loss for multiple batches
def estimate_loss():
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



class Head(nn.Module):


    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        #tril - buffer  - lower triangular matrix 
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        B,T, C = x.shape
        k = self.key(x) #(B, T, C)
        q = self.query(x) # (B, T, C
        
        #compute attention scores "affinities"
        #communication betweeen keys and queriies - normalize it 
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B, T, C) @ (B, C, T) -> (B, T, T)
        #for all elements where trill is equal to zero make -inf
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T)
        #Softmax - normalization operation - exponatiate every entry and divide by sum
        wei = F.softmax(wei, dim = -1) #(B, T, T)
        wei = self.dropout(wei)

        #perform weighted aggregation of the values  
        # v = values we aggregate instead of raw outputs

        v = self.value(x) # (B, T, C)
        out = wei @ v #(B, T, T) @ (B, T, C) -> (B, T, C)
        return out 


class MultiHeadAttention(nn.Module):
   # multiple heads of self-attention in parallel

   def __init__(self, num_heads, head_size):
      super().__init__()
      #create multiple heads 
      self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
      self.proj = nn.Linear(head_size * num_heads, n_embd)
      self.dropout = nn.Dropout(dropout)

    #run in parallel and concatenate
   def forward(self, x):
      out = torch.cat([h(x) for h in self.heads], dim=-1) #output of self-attention
      out = self.dropout(self.proj(out)) #projectioni is linear transformatio of outcome of 'out'
      #projection back into residual
      return out


class FeedForward(nn.Module):
   #self- attention = communcation, feedforward = think on the data gathered 
   #linear layer followed by non-linearity

   def __init__(self, n_embd):
      super().__init__()
      self.net = nn.Sequential(
         #multiplier of 4 
         nn.Linear(n_embd, 4*n_embd),
         nn.ReLU(),
         nn.Linear(4*n_embd, n_embd),
         nn.Dropout(dropout),
      )
   def forward(self, x):
      return self.net(x)

class Block(nn.Module):
   #transformer block: communication followed by computation

   def __init__(self, n_embd, n_head):
      #n_embd: embedding dimension, n_head - number of heads we'd like
      super().__init__()
      head_size = n_embd // n_head #n_embd - 32 and head_size should be 8 so n_head = 4
      self.sa = MultiHeadAttention(n_head, head_size) #commmunication
      self.ffwd = FeedForward(n_embd) #computation
      self.ln1 = nn.LayerNorm(n_embd) #normalizing features 
      self.ln2 = nn.LayerNorm(n_embd)
   def forward(self, x):
      #residual connections - optimization 
      x = x+self.sa(self.ln1(x))
      x = x+self.ffwd(self.ln2(x))
      return x 

class BigramLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()

    #input will refer to embedding table and pluck out row of table corresponding to index 
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm 
    #dropout - randomly shuts off some set of neurons and trains w/o them
        #dropout - regularization
    #self.sa_heads = MultiHeadAttention(4, n_embd//4) # 4 heads of 8-dimensional self-attention
    #self.ffwd = FeedForward(n_embd)
    #self.sa_head = Head(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  #idx = index (inputs x)
  #token embedding table
  def forward(self, idx, targets=None):

    B, T = idx.shape
    #idx and targets are both (B,T) tensor of integers

    tok_emb = self.token_embedding_table(idx) #(B, T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device = device)) #(T, C)
    #x holds token identities and positions where they occur 
    x = tok_emb + pos_emb #(B, T, C)
    #x = self.sa_heads(x) # apply one head of self-attention (B, T, C)
    #x = self.ffwd(x) #(B, T, C)
    x = self.blocks(x)
    x = self.ln_f(x) #(B, T, C)
    logits = self.lm_head(x) #(B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    #idx is (B, T) array of indicies in the current context
    for _ in range(max_new_tokens):
      #crop ix to last block_size tokens
      idx_cond = idx[:, -block_size:]

      logits, loss = self(idx_cond)
   
      logits= logits[:, -1, :]
  
      probs = F.softmax(logits, dim = -1)
    
      idx_next = torch.multinomial(probs, num_samples=1)
    
      idx = torch.cat((idx, idx_next), dim=1)
    return idx
model = BigramLanguageModel()
m = model.to(device)

#train model
#create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)

for iter in range(max_iters):
  
  #evaluate loss on train and val sets sometimes
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
  
  #sample batch of data
  xb, yb = get_batch('train')

  #evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

#generate from model
#create batch - 1 x 1 tensor holding a zero (type is integer)
#ask for 100 tokens 
#conver to list 
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))



from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import math

# vocab_size: defines vocabulary size which in our case is 50257 tokens
# context_length: represents the model's maximum input token length, constrained by the positional embeddings
# emb_dim: refers to the embedding size for token inputs, where each input token is converted into a 768 dimensional vector
# n_heads: specifies the number of attention heads in the multi-head attention mechanism
# n_layers: indicates the number of transformer blocks in the model
# drop_rate: determines the intensity of the dropout mechanism; 0.1 means 10% of hidden units are dropped during training to reduce overfitting
# qkv_bias: contraols whether a bias vector is added in the Linear layers of the multi-head attention mechanism when computing the query, key, and value tensors; by default we disable, following common practice in LLMs


# token embeddings: responsible for transforming input tokens into continuous numerical vectors that model can process
#       - they capture the semantic meaning of the tokens
#       - crucial first step in how model represents and understands the language
# embedding takes a token and returns a vector
# there are vocab_size different tokens that can be passed and each token maps to a emb_dim dimensional vector


# positional encoding: used to provide model with info about order of tokens in a sequence, since the model itself does not have a built-in way to understand the position of tokens
#       - transformer architecture processes tokens in parallel, therefore explicit information about token positions is requires to preserve the notion of word order
# 1. Positional Encoding in Transformers: transformer architecture uses positional encodings to inject info about position of each token in the sequence into input embeddings
#       - this is done by adding a fixed positional encoding vector to each token's word embedding
# 2. Sinusoidal Encoding: GPT2 uses sinusoidal positional encodings. The positional encodings are calculated based on the position, pos, of the token and dimension, i, of the encoding vector
# Why sinusoidal functions?
# - chosen because they allow model to generalize to sequence lengtsh longer than those seen during training
# - Periodicity of sine and cosine functions enables the model to learn relationships between different positions
# - smooth continuous nature of these functions give model some geometric structure for recognizing positional patterns



@dataclass
class GPT_config:
    vocab_size: int = 50304         # vocabulary size
    context_length: int = 1024      # context length
    n_layer: int = 12               # number of layers  
    n_head: int = 12                # number of attention heads
    emb_dim: int = 768              # embedding dimension
    dropout: float = 0.1            # dropout rate
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster



class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        # initialize the embedding layer with the specified vocab size and embedding dimension
        self.embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        # forward pass: convert input token IDs to their corresponding embeddings
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_length=512):
        super().__init__()
        # initialize tensor to hold positional encodings
        pos_enc = torch.zeros(max_seq_length, embed_size)
    
        # create tensor for positions (0 to max_seq_length)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # calculate the division term for the sine and cosine functions
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))

        # apply sine to even indices and cosine to odd indices
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        
        # register the positional encodings as a buffer (not a model parameter)
        self.register_buffer('pos_enc', pos_enc.unsqueeze(0))   # shape: (1, max_seq_length, embed_size)

    def forward(self, x):
        # add the positional encodings to the input embeddings
        return x + self.pos_enc[:, :x.size(1)]


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, qkv_bias=False):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.key = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.value = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        batch_size = x.shape[0]

        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention = (q @ k.transpose(-1, -2)) * self.head_dim**-0.5 
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)

        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        return self.out(out)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super().__init__()

        # first linear layer that transforms input from embedding size to hidden size
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        
        # second linear layer that transforms from hidden size back to embedding size
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        
        # gelu non-linearity 
        self.gelu = nn.GELU()

    def forward(self, x):
        # apply first linear layer, then non-linearity, then second linear layer
        return self.fc2(self.gelu(self.fc1(x)))


# transformer block in gpt2 model processes input text through sequence of steps:
#       - applies causal multi-head self attention, where each token focuses on previous tokens in the sequence to capture dependencies
#       - output from this is passed through a feed-forward network that transforms the data to capture more complex patterns
#       - both stages are followed by residual connections and layer normalization to ensure stability and efficient learning
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1, qkv_bias=False):
        super().__init__()

        # initialize multi-head attention layer
        self.mha = MultiHeadAttention(embed_size, num_heads, qkv_bias)

        # initialize feed-forward network
        self.ff = FeedForward(embed_size, ff_hidden_size)

        # initialize layer norm for attention output
        self.ln1 = nn.LayerNorm(embed_size)

        # initialize layer norm for feed-forward output
        self.ln2 = nn.LayerNorm(embed_size)

        # initialize dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # apply multi-head attention and add residual connection, followed by layer norm
        attention_output = self.ln1(x + self.dropout(self.mha(x, mask)))
        
        # apply feed-forward network, add residual connection, followed by layer norm
        ff_output = self.ln2(attention_output + self.dropout(self.ff(attention_output)))

        return ff_output


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        # initialize embedding layer to convert token IDs to embeddings
        self.embedding = Embedding(config["vocab_size"], config["emb_dim"])

        # initialize positional encoding to add positional information to embeddings
        self.positional_encoding = PositionalEncoding(config["emb_dim"], config["context_length"])

        # create list of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            # each transformer block consists of multi-head attention and feed-forward layers
            TransformerBlock(config["emb_dim"], config["n_heads"], config["emb_dim"] * 4, config["drop_rate"], config["qkv_bias"])
            for _ in range(config["n_layers"])  # repeat for number of layers 
        ])
        
        # final linear layer to project output back to vocab size for logits
        self.fc_out = nn.Linear(config["emb_dim"], config["vocab_size"])

        # dropout layer for regularization
        self.dropout = nn.Dropout(config["drop_rate"])
    
    
    def forward(self, x, mask=None):
        # convert input token IDs to embeddings and add positional encodings
        x = self.dropout(self.positional_encoding(self.embedding(x)))

        # pass embeddings through each transformer block
        for block in self.transformer_blocks:
            x = block(x, mask)      # apply transformer block with optional mask
        
        # project final output to vocab size
        return self.fc_out(x)       

    

def generate(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):

        # crop current context if exceeds supported context size
        idx_cond = idx[:, -context_size:]
        
        # get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # focus only on last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)   # (batch, vocab_size)

        # get the idx of the vocab entry with highest probability value
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)    # (batch, 1)
        
        # append sampled index to running sequence
        idx = torch.cat((idx, idx_next), dim=1)     # (batch, n_tokens+1)
    
    return idx


model = GPT(GPT_config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
start_context = " "

encoded = tokenizer.encode(start_context)
print(f"encoded:{encoded}")

encoded = torch.tensor(encoded).unsqueeze(0)
model.eval()

out = generate(model=model, idx=encoded, max_new_tokens=20, context_size=GPT_config["context_length"])

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)













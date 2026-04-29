import torch
import torch.nn as nn

### Self-Attention (Without Weights) ###
import torch
#Step 1
word_embeddings = torch.tensor(
  [[0.32, 0.68, 0.45], # The      (x^1)
   [0.71, 0.23, 0.89], # chef     (x^2)
   [0.55, 0.92, 0.37], # prepared (x^3)
   [0.18, 0.79, 0.60], # a        (x^4)
   [0.84, 0.41, 0.13], # delicious(x^5)
   [0.29, 0.63, 0.76], # meal     (x^6)
   [0.50, 0.15, 0.95], # and      (x^7)
   [0.67, 0.38, 0.82], # it       (x^8)
   [0.43, 0.91, 0.26], # was      (x^9)
   [0.75, 0.20, 0.58], # served   (x^10)
   [0.36, 0.72, 0.49], # with     (x^11)
   [0.88, 0.54, 0.11]] # wine     (x^12)
)

# Step 2 Calculate attention scores (Pairwise compare)
attn_scores = word_embeddings @ word_embeddings.T

# Step 3 Calculate attention weights (Normalize)
attn_weights = torch.softmax(attn_scores, dim=-1)

# Step 4 Calculate context vectors
all_context_vecs = attn_weights @ word_embeddings

torch.manual_seed(1240)

d_in = word_embeddings.shape[1]
d_out = 2

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # linear projections for query, key, and value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
        # output projection
        self.out_proj = nn.Linear(d_out, d_out)

        # dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
        # causal mask to prevent attending to future tokens
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape       # batch size, num_tokens: sequence_length

        q = self.W_query(x)                 # shape (b, num_tokens, d_out)
        k = self.W_key(x)                   # shape (b, num_tokens, d_out)
        v = self.W_value(x)                 # shape (b, num_tokens, d_out)

        # reshape for multi-head attention
        q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose for attention computation
        # (b, num_tokens, num_heads, head_dim) --> (b, num_heads, num_tokens, head_dim)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # compute scaed dot-product attention
        attn_scores = q @ k.transpose(2, 3) # shape (b, num_heads, num_tokens, num_tokens)

        # apply causal mask
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # compute attention weights
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # apply attention weights to values
        context_vectors = attn_weights @ v          # shape (b, num_heads, num_tokens, head_dim)

        # reshape and combine heads
        context_vectors = context_vectors.transpose(1, 2).contiguous()      # shape (b, num_tokens, num_heads, head_dim)
        context_vectors = context_vectors.view(b, num_tokens, self.d_out)   # shape (b, num_tokens, d_out)
        
        context_vectors = self.out_proj(context_vec)

        return context_vectors


# Initialize the SelfAttention module
torch.manual_seed(1240)  # Reset seed for consistent results
self_attention = SelfAttention(d_in, d_out)

# Apply self-attention to word embeddings
result = self_attention(word_embeddings)
print(result)

import math
import torch
import torch.nn as nn

class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dims // num_heads
        
        # Initialize projections
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)
        
        # Initialize RoPE (Rotary Position Embedding)
        self.rope = RoPE(head_dim)
        
        # Scaling factor for attention scores
        self.scale = math.sqrt(1.0 / head_dim)

    def forward(self, queries, keys, values, mask=None, cache=None):
        # Project inputs
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Get shapes
        batch_size, seq_len, _ = queries.shape
        head_dim = queries.shape[-1] // self.num_heads

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, head_dim)
        keys = keys.view(batch_size, seq_len, self.num_heads, head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, head_dim)

        # Transpose for attention computation
        queries = queries.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Apply RoPE and handle cache if provided
        if cache is not None:
            key_cache, value_cache = cache
            offset = key_cache.size(2)  # Get current position for RoPE
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys = torch.cat([key_cache, keys], dim=2)
            values = torch.cat([value_cache, values], dim=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Compute attention scores
        scores = torch.matmul(queries * self.scale, keys.transpose(-2, -1))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute weighted sum
        output = torch.matmul(attention_weights, values)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, -1)
        output = self.out_proj(output)

        return output, (keys, values)

class RoPE(nn.Module):
    def __init__(self, dim: int, traditional: bool = True):
        super().__init__()
        self.dim = dim
        self.traditional = traditional
        
    def forward(self, x: torch.Tensor, offset: int = 0):
        shape = x.shape
        x = x.view(-1, shape[-2], shape[-1])
        
        # Generate position encodings
        position = torch.arange(offset, offset + x.shape[-2], device=x.device).unsqueeze(1)
        
        # Generate frequency bands
        freq = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device) * -(math.log(10000.0) / self.dim)
        )
        
        # Compute angles
        angles = position * freq
        
        # Apply rotary embeddings
        x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        angles = angles.unsqueeze(0).expand(x_complex.shape[:-1] + (angles.shape[-1],))
        
        if self.traditional:
            x_rotated = x_complex * torch.exp(-1j * angles)
        else:
            x_rotated = x_complex * (torch.cos(angles) - 1j * torch.sin(angles))
        
        x_out = torch.view_as_real(x_rotated).reshape(*shape)
        return x_out.view(*shape)
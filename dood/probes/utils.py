import torch

class MultiHeadAttentionExplicitQKV(torch.nn.Module):
    class QKV(torch.nn.Module):
        def forward(self, query, key, value, original_mha):
            q, k, v = torch.nn.functional._in_projection_packed(query, key, value, original_mha.in_proj_weight, original_mha.in_proj_bias)
            return q, k, v

    def __init__(self, mha):
        super().__init__()
        self.mha = mha
        self.explicit_qkv = self.QKV()

    def forward(self, query, key, value, **kwargs):
        q, k, v = self.explicit_qkv(query, key, value, self.mha)
        return self.mha(query, key, value, **kwargs)
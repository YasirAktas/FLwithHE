"""Encryption abstraction layer.
Provides a PlainContext (no encryption) and a HomomorphicContext using TenSEAL CKKS.
"""
from typing import Any, Tuple
import torch
try:
    import tenseal as ts
except ImportError:  # TenSEAL is optional; install it to use HomomorphicContext
    ts = None

class PlainContext:
    def encrypt(self, tensor):
        return tensor  # no-op

    def decrypt(self, tensor):
        return tensor  # no-op

    def add(self, a, b):
        return a + b

    def mul_scalar(self, a, s: float):
        return a * s

class HomomorphicContext:
    def __init__(self, poly_modulus_degree: int = 8192, coeff_mod_bit_sizes: Tuple[int, ...] = (60, 40, 40, 60), global_scale: float = 2 ** 40):
        if ts is None:
            raise ImportError("TenSEAL not installed. Please `pip install tenseal` to use HomomorphicContext.")
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, list(coeff_mod_bit_sizes))
        self.ctx.global_scale = global_scale
        self.ctx.generate_galois_keys()
        # CKKS offers poly_modulus_degree/2 slots for real numbers
        self.n_slots = poly_modulus_degree // 2

    class EncryptedTensor:
        def __init__(self, cts, shape):
            # cts: list of ckks_vector chunks
            self.cts = cts
            self.shape = shape

    def encrypt(self, tensor: torch.Tensor):
        flat = tensor.detach().cpu().view(-1).tolist()
        cts = []
        for i in range(0, len(flat), self.n_slots):
            chunk = flat[i:i + self.n_slots]
            cts.append(ts.ckks_vector(self.ctx, chunk))
        return HomomorphicContext.EncryptedTensor(cts, tuple(tensor.size()))

    def decrypt(self, enc: "HomomorphicContext.EncryptedTensor"):
        vals = []
        for ct in enc.cts:
            vals.extend(ct.decrypt())
        t = torch.tensor(vals, dtype=torch.float32).view(enc.shape)
        return t

    def add(self, a: "HomomorphicContext.EncryptedTensor", b: "HomomorphicContext.EncryptedTensor"):
        if a.shape != b.shape:
            raise ValueError("Shape mismatch in encrypted add")
        if len(a.cts) != len(b.cts):
            raise ValueError("Chunk mismatch in encrypted add")
        cts = [ca + cb for ca, cb in zip(a.cts, b.cts)]
        return HomomorphicContext.EncryptedTensor(cts, a.shape)

    def mul_scalar(self, a: "HomomorphicContext.EncryptedTensor", s: float):
        cts = [ca * s for ca in a.cts]
        return HomomorphicContext.EncryptedTensor(cts, a.shape)

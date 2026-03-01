"""Encryption abstraction layer.
Provides:
- PlainContext: no encryption
- HomomorphicContext: fully homomorphic CKKS via TenSEAL
- PaillierContext: additive partial HE via Paillier (integer-only scheme)
"""
from typing import Any, Tuple
import torch
try:
    import tenseal as ts
except ImportError:  # TenSEAL is optional; install it to use HomomorphicContext
    ts = None

try:
    from phe import paillier
except ImportError:  # Paillier is optional; install it to use PaillierContext
    paillier = None

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


class PaillierContext:
    """Additively homomorphic encryption using Paillier.

    This is a *partial* HE scheme: it supports
    - ciphertext addition
    - multiplication of a ciphertext by a *plaintext integer* scalar

    Since Paillier is integer-only, tensors are quantized using a fixed
    scaling factor before encryption and de-quantized after decryption.
    """

    def __init__(self, key_length: int = 1024, scale: float = 1e4):
        if paillier is None:
            raise ImportError("Paillier library 'phe' not installed. Please `pip install phe` to use PaillierContext.")
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.scale = float(scale)
        # Hint for callers (e.g., Aggregator) that this context expects
        # integer scalars in mul_scalar.
        self.scalar_mode = "int"

    class EncryptedTensor:
        def __init__(self, cts, shape):
            # cts: flat list of Paillier ciphertexts
            self.cts = cts
            self.shape = shape

    def _encode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a floating tensor to int64 using fixed-point scaling."""
        return torch.round(tensor * self.scale).to(torch.int64)

    def _decode(self, tensor_int: torch.Tensor) -> torch.Tensor:
        """De-quantize an int tensor back to float32."""
        return tensor_int.to(torch.float32) / self.scale

    def encrypt(self, tensor: torch.Tensor):
        # Move to CPU and flatten
        quantized = self._encode(tensor.detach().cpu())
        flat = quantized.view(-1).tolist()
        cts = [self.public_key.encrypt(int(v)) for v in flat]
        return PaillierContext.EncryptedTensor(cts, tuple(tensor.size()))

    def decrypt(self, enc: "PaillierContext.EncryptedTensor"):
        vals = [self.private_key.decrypt(ct) for ct in enc.cts]
        t_int = torch.tensor(vals, dtype=torch.int64).view(enc.shape)
        return self._decode(t_int)

    def add(self, a: "PaillierContext.EncryptedTensor", b: "PaillierContext.EncryptedTensor"):
        if a.shape != b.shape:
            raise ValueError("Shape mismatch in Paillier encrypted add")
        if len(a.cts) != len(b.cts):
            raise ValueError("Chunk mismatch in Paillier encrypted add")
        cts = [ca + cb for ca, cb in zip(a.cts, b.cts)]
        return PaillierContext.EncryptedTensor(cts, a.shape)

    def mul_scalar(self, a: "PaillierContext.EncryptedTensor", s: Any):
        """Multiply encrypted tensor by an *integer* scalar.

        For Paillier, homomorphic multiplication is only defined for
        integer plaintext scalars. Callers are expected to pass integers
        (e.g., number of samples); any float will be cast with int().
        """
        k = int(s)
        cts = [ct * k for ct in a.cts]
        return PaillierContext.EncryptedTensor(cts, a.shape)

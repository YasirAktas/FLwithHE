"""Encryption abstraction layer.
Currently provides a PlainContext (no encryption) and a stub HomomorphicContext template.
Extend HomomorphicContext with a real library (e.g., TenSEAL) later.
"""
from typing import Any

class PlainContext:
    def encrypt(self, tensor):
        return tensor  # no-op

    def decrypt(self, tensor):
        return tensor  # no-op

    def add(self, a, b):
        return a + b

class HomomorphicContext:
    def __init__(self):
        # Placeholder: set up keys or context when using a real HE lib
        self.backend: Any = None

    def encrypt(self, tensor):
        # Replace with real encryption call
        return tensor

    def decrypt(self, tensor):
        # Replace with real decryption call
        return tensor

    def add(self, a, b):
        # Replace with encrypted addition
        return a + b

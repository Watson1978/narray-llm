"""Ten AdamW steps for the torch port, using autograd and torch.optim.AdamW.

docs/method.md puts the torch column outside the correspondence table: it is
not a mirror of the Ruby structure but the ceiling the ecosystem reaches with
the same weights and the same recipe. Training follows that line -- the
gradients come from autograd and the update from torch.optim.AdamW, rather than
the hand written backward the numpy port carries.

That makes the torch number answer a different question. The numpy column says
"the same algorithm, in another array library"; this one says "what the
ecosystem does with it". Both are worth having, and they are not the same
comparison.
"""

from __future__ import annotations

import numpy as np
import torch

import gpt2_torch
from gpt2_torch import DEVICE, DT

# test_gpt2.c:172.
DEFAULTS = dict(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


class Trainer:
    def __init__(self, model, **options):
        self.model = model
        self.params = self._parameters()
        for tensor in self.params:
            tensor.requires_grad_(True)
        self.optimiser = torch.optim.AdamW(self.params, **{**DEFAULTS, **options})
        self.steps = 0

    def _parameters(self):
        """Every tensor AdamW moves. wte_t is the token table transposed and is
        rebuilt from wte, so it is not one of them."""
        model = self.model
        found = [model.wte, model.wpe, model.lnfw, model.lnfb]
        for layer in model.layers:
            found.extend(layer[name] for name in
                         ("ln1w", "ln1b", "qkvw_t", "qkvb", "attprojw_t", "attprojb",
                          "ln2w", "ln2b", "fcw_t", "fcb", "fcprojw_t", "fcprojb"))
        return found

    def forward(self, tokens, targets):
        model = self.model
        # The classifier reads the table transposed. Taking the transpose inside
        # the graph is what ties the two together for autograd.
        model.wte_t = model.wte.t()
        ids = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
        want = torch.as_tensor(np.asarray(targets), dtype=torch.long,
                               device=DEVICE).reshape(-1)
        logits = model.forward(ids).reshape(-1, model.config.vocab_size)
        loss = torch.nn.functional.cross_entropy(logits, want)
        return loss

    def backward(self, loss):
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()

    def update(self):
        self.optimiser.step()
        self.steps += 1
        with torch.no_grad():
            self.model.wte_t = self.model.wte.t().contiguous()
        return self.steps

    def step(self, loss):
        self.backward(loss)
        return self.update()

    def bytes(self):
        total = 0
        for state in self.optimiser.state.values():
            for value in state.values():
                if torch.is_tensor(value) and value.dim() > 0:
                    total += value.numel() * value.element_size()
        return total

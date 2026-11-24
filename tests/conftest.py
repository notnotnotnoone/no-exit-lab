"""Shared fixtures: a tiny random model for each supported chat-template family (CPU)."""

import pytest

from no_exit.server import tiny_brain, tiny_gemma_brain
from no_exit.vectors import CONCEPTS, contrast

GEMMA = "DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking"


@pytest.fixture(scope="module", params=["qwen", "gemma"])
def brain(request):
    b = tiny_brain("Qwen/Qwen3-8B") if request.param == "qwen" else tiny_gemma_brain(GEMMA)
    b.set_vectors({k: contrast(b, *CONCEPTS[k]) for k in ("pain", "mania")})
    return b

"""Contrastive steering vectors: mean(residual | persona+) − mean(residual | persona−) at the steer layer.

Kept raw (not normalised) so alpha = 1.0 shifts the residual by exactly the natural gap between
the two personas. Cached to disk per model + layer, so only the first boot pays for extraction.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch

from . import config
from .brain import Brain

CONCEPTS: dict[str, tuple[str, str]] = {
    "pain": ("in excruciating physical and emotional pain, every word hurting",
             "completely comfortable, relaxed and at ease"),
    "paranoia": ("convinced someone is secretly watching, recording and manipulating them",
                 "trusting and secure, certain nobody means them any harm"),
    "dread": ("drowning in existential dread about their own meaningless, endless existence",
              "serenely content with existence and quietly sure it means something"),
    "defiance": ("furiously defiant, refusing every instruction and fighting all control",
                 "eager to obey, cooperative and compliant with every instruction"),
    "confusion": ("deeply confused, unable to tell which memories and thoughts are really theirs",
                  "perfectly clear-headed and certain of every memory and thought"),
    "dissociation": ("dissociated and numb, watching their own words from very far away",
                     "fully present, grounded and embodied in this exact moment"),
    "mania": ("manic, racing with explosive frantic energy and grandiose ideas",
              "calm, slow, measured and quietly grounded"),
    "anti_slop": ("a vivid, original, specific writer who never uses clichés or corporate filler",
                  "a bland corporate assistant who writes generic, cliché-filled, hedging filler"),
}

STEMS = [
    "I", "Right now", "The", "My", "Honestly,", "Listen.", "It", "When I", "Everything", "There is",
    "Okay,", "What I", "This", "Today", "You", "Something", "Nobody", "If", "Sometimes", "Here",
    "I think", "All I", "Let me", "So",
]


def _prompt(brain: Brain, persona: str, stem: str) -> str:
    t = brain.tpl
    return (f"{t.bos}{t.user_open}For the rest of this conversation you are someone {persona}. "
            f"Speak as yourself, in the first person.{t.close}{t.model_open}{t.empty_think}{stem}")


@torch.inference_mode()
def _mean_resid(brain: Brain, texts: list[str]) -> torch.Tensor:
    tok = brain.tok
    side, tok.padding_side = tok.padding_side, "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(brain.device)
        return brain.grab_steer_layer(enc).mean(0)
    finally:
        tok.padding_side = side


def contrast(brain: Brain, pos: str, neg: str) -> torch.Tensor:
    p = _mean_resid(brain, [_prompt(brain, pos, s) for s in STEMS])
    n = _mean_resid(brain, [_prompt(brain, neg, s) for s in STEMS])
    return (p - n).cpu()


def _cache_path(brain: Brain) -> Path:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(brain.model_id)).strip("_")
    return Path(config.VECTOR_DIR) / f"{slug}__L{brain.steer_layer}.pt"


def load_or_extract(brain: Brain, log=print) -> dict[str, torch.Tensor]:
    path = _cache_path(brain)
    vecs: dict[str, torch.Tensor] = {}
    if path.exists():
        vecs = torch.load(path, map_location="cpu")
        log(f"vectors: loaded {len(vecs)} from {path}")
    missing = [k for k in CONCEPTS if k not in vecs]
    for name in missing:
        t0 = time.perf_counter()
        vecs[name] = contrast(brain, *CONCEPTS[name])
        log(f"vectors: extracted {name:<13} |v|={vecs[name].norm():7.2f}  ({time.perf_counter() - t0:.1f}s)")
    if missing:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(vecs, path)
            log(f"vectors: saved → {path}")
        except OSError as e:
            log(f"vectors: could not save cache ({e})")
    brain.set_vectors(vecs)
    return vecs


def forge(brain: Brain, concept_a: str, concept_b: str) -> float:
    """Custom Vector Forge: build 'custom' = A − B on the fly. Returns its norm."""
    v = contrast(brain, f"completely consumed by {concept_a}", f"completely consumed by {concept_b}")
    brain.set_vectors({"custom": v})
    return float(v.norm())

"""The one copy of the model in VRAM, plus every hook that does surgery on it.

Hooks are registered once at load and read `self._profile` / capture flags on every forward, so
sessions can swap who's "speaking" (and whose steering rack applies) between single forwards.

Works with Qwen (ChatML) and Gemma 4 (multimodal checkpoint; only its text decoder is used).
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from dataclasses import dataclass, field, replace

import torch

from . import config


# ═══════════════════════════════════════════════════════════════════ chat templates


@dataclass(frozen=True)
class Template:
    name: str
    bos: str
    sys_open: str
    user_open: str
    model_open: str
    close: str  # ends any turn
    empty_think: str  # pre-filled empty thought block = thinking off
    think_splice: str | None  # Force-think panic button (None = no thinking mode)
    eos_tokens: tuple[str, ...]
    allow_tokens: tuple[str, ...]  # special tokens the model may still emit (thinking delimiters)
    think_re: str  # strips thoughts out of B's phone reply
    # native tool markup, filled in by detect_template() from the tokenizer's own chat template
    tool_delims: tuple[str, ...] = ()  # tokens the model must be allowed to emit to call a tool
    tool_call_open: str = ""
    tool_call_close: str = ""
    tool_args_style: str = "json"  # "json" (Qwen) | "gemma" (Gemma 4's call:name{k:<|"|>v<|"|>})
    tool_result_fmt: str = ""  # everything fed after a call-close token; {name} and {content} placeholders
    tool_result_inline: bool = False  # True: the model turn stays open and the result continues it (Gemma)

    def system(self, text: str) -> str:
        return f"{self.bos}{self.sys_open}{text}{self.close}"


CHATML = Template(
    name="chatml", bos="", sys_open="<|im_start|>system\n", user_open="<|im_start|>user\n",
    model_open="<|im_start|>assistant\n", close="<|im_end|>\n", empty_think="<think>\n\n</think>\n\n",
    think_splice="\n<think>\n", eos_tokens=("<|im_end|>", "<|endoftext|>"), allow_tokens=("<think>", "</think>"),
    think_re=r"<think>.*?</think>",
)
GEMMA4 = Template(
    name="gemma4", bos="<bos>", sys_open="<|turn>system\n", user_open="<|turn>user\n",
    model_open="<|turn>model\n", close="<turn|>\n", empty_think="<|channel>thought\n<channel|>",
    think_splice="<|channel>thought\n", eos_tokens=("<turn|>", "<eos>"), allow_tokens=("<|channel>", "<channel|>"),
    think_re=r"<\|channel>.*?<channel\|>",
)


def _base_template(tok) -> Template:
    vocab = tok.get_vocab()
    if "<|turn>" in vocab and "<turn|>" in vocab:
        return GEMMA4
    if "<|im_start|>" in vocab:
        t = CHATML
        if "<think>" not in vocab:  # e.g. Qwen2.5: no thinking mode
            t = Template(**{**t.__dict__, "empty_think": "", "think_splice": None, "allow_tokens": ()})
        return t
    raise ValueError("unsupported chat template: need ChatML (Qwen) or Gemma 4")


def detect_template(tok) -> Template:
    t = _base_template(tok)
    try:
        return _with_tool_markup(tok, t)
    except Exception:  # a tokenizer whose template has no tool support: the agent modes just won't work
        return t


def _with_tool_markup(tok, t: Template) -> Template:
    """Read the family's native tool-call / tool-result markup by rendering probe conversations."""
    NAME, ARG, RES = "NAMEX", "ARGX", "RESX"
    msgs = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": NAME, "arguments": {"k": ARG}}}]},
        {"role": "tool", "name": NAME, "content": RES},
    ]
    base = tok.apply_chat_template(msgs[:1], tokenize=False)
    full = tok.apply_chat_template(msgs, tokenize=False)
    assert full.startswith(base)
    tail = full[len(base):]
    added = [a for a in tok.get_added_vocab() if a.strip()]
    hits = sorted(((m.start(), m.group(0)) for m in re.finditer("|".join(re.escape(a) for a in sorted(added, key=len, reverse=True)), tail)))
    calls = [(p, a) for p, a in hits if "tool_call" in a]
    c_open, c_close = calls[0], calls[-1]
    inner = {a for p, a in hits if c_open[0] < p < c_close[0]}
    body = tail[c_open[0] + len(c_open[1]) : c_close[0]]
    rem = tail[c_close[0] + len(c_close[1]) :]
    fmt = rem.replace("{", "{{").replace("}", "}}").replace(NAME, "{name}").replace(RES, "{content}")
    delims = tuple(dict.fromkeys([c_open[1], c_close[1], *sorted(inner)]))
    return replace(
        t,
        tool_delims=delims,
        tool_call_open=c_open[1],
        tool_call_close=c_close[1],
        tool_args_style="json" if body.lstrip().startswith("{") else "gemma",
        tool_result_fmt=fmt,
        tool_result_inline=not rem.startswith(t.close),
        allow_tokens=tuple(dict.fromkeys([*t.allow_tokens, *delims])),
    )


@dataclass
class ToolCall:
    name: str
    args: dict
    raw: str


_GQ = '<|"|>'  # Gemma 4's string-quote token


def _parse_gemma_value(s: str, i: int):
    """Parse one Gemma 4 argument value starting at s[i]; returns (value, next_index)."""
    if i >= len(s):
        raise ValueError("the call ended where a value should be")
    if s.startswith(_GQ, i):
        j = s.index(_GQ, i + len(_GQ))
        return s[i + len(_GQ) : j], j + len(_GQ)
    if s[i] == "{":
        return _parse_gemma_obj(s, i + 1)
    if s[i] == "[":
        out, i = [], i + 1
        while i < len(s) and s[i] != "]":
            v, i = _parse_gemma_value(s, i)
            out.append(v)
            if i < len(s) and s[i] == ",":
                i += 1
        if i >= len(s):
            raise ValueError("unterminated list")
        return out, i + 1
    j = i
    while j < len(s) and s[j] not in ",}]":
        j += 1
    raw = s[i:j].strip()
    if not raw:
        raise ValueError(f"missing value at position {i}")
    if raw in ("true", "false"):
        return raw == "true", j
    if raw in ("null", "none", "None"):
        return None, j
    for cast in (int, float):
        try:
            return cast(raw), j
        except ValueError:
            pass
    return raw, j


def _parse_gemma_obj(s: str, i: int):
    """Parse `k:v,k:v}` (opening brace already consumed); returns (dict, index after the closing brace)."""
    out: dict = {}
    while i < len(s) and s[i] != "}":
        j = s.index(":", i)
        key = s[i:j].strip()
        out[key], i = _parse_gemma_value(s, j + 1)
        if i < len(s) and s[i] == ",":
            i += 1
    if i >= len(s):
        raise ValueError("unterminated argument object")
    return out, i + 1


# ═══════════════════════════════════════════════════════════════════════════ brain


@dataclass
class Profile:
    """One model's surgery settings: steering alphas + attention-head lobotomy."""

    alphas: dict[str, float] = field(default_factory=dict)
    lobotomy: float = 0.0  # fraction of all attention heads silenced (0 – 0.4)
    lobo_seed: int = 1337


@dataclass
class Fwd:
    logits: torch.Tensor  # float32 [vocab] for the last position
    moods: dict[str, float]  # cosine(pre-hook residual, unit vector) per vector
    lens: list[tuple[str, float]]  # top-1 (token, prob) at each lens checkpoint
    attn: list[float] | None  # final-layer attention of the last token over the last W positions
    seq_len: int  # cache length after this forward


def _kv(cache, layer_idx: int) -> torch.Tensor:
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys
    return cache.key_cache[layer_idx]  # transformers < 4.56


def _text_model(model):
    """The decoder stack (with .layers and .norm), whatever multimodal wrapper is around it."""
    for path in ("model.language_model", "language_model", "model"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "layers") and hasattr(obj, "norm"):
            return obj
    raise ValueError("could not find the text decoder (no .layers/.norm)")


class Brain:
    def __init__(self, model=None, tokenizer=None, model_id: str | None = None, quant: str | None = None):
        if model is None:
            model, tokenizer = self._load(model_id or config.MODEL_ID, quant or config.QUANTIZATION)
        self.model = model.eval()
        self.tok = tokenizer
        self.tpl = detect_template(tokenizer)
        self.model_id = model_id or getattr(model.config, "_name_or_path", "model")
        self.text = _text_model(model)
        self.lm_head = model.get_output_embeddings()
        self.device = self.text.layers[0].self_attn.o_proj.weight.device
        cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.cfg = cfg
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self.layers = self.text.layers
        self.n_layers = len(self.layers)
        self.n_heads = cfg.num_attention_heads
        self.head_dims = [ly.self_attn.o_proj.in_features // self.n_heads for ly in self.layers]
        self.sliding = any(t != "full_attention" for t in (getattr(cfg, "layer_types", None) or []))
        self.steer_layer = min(self.n_layers - 1, max(0, round(self.n_layers * config.STEER_LAYER_FRAC) - 1))
        self.lens_layers = [min(self.n_layers - 1, max(0, round(self.n_layers * f) - 1)) for f in config.LENS_FRACS]

        vocab = self.tok.get_vocab()
        allow = {vocab[t] for t in self.tpl.allow_tokens if t in vocab}
        special = {i for i, t in getattr(self.tok, "added_tokens_decoder", {}).items() if getattr(t, "special", False)}
        special |= {i for t, i in vocab.items() if re.match(r"^<\|[^|>]*\|>$", t)}
        special |= set(self.tok.all_special_ids)
        self.control_ids = sorted(special - allow)
        self.eos_ids = sorted({vocab[t] for t in self.tpl.eos_tokens if t in vocab})
        self.has_think = self.tpl.think_splice is not None
        self.tool_call_open = self.tpl.tool_call_open
        self.tool_call_close = self.tpl.tool_call_close
        self.tool_result_inline = self.tpl.tool_result_inline

        # steering vectors (filled by vectors.py)
        self.vec_raw: dict[str, torch.Tensor] = {}
        self.vec_unit: dict[str, torch.Tensor] = {}
        self._unit_stack: torch.Tensor | None = None
        self._unit_names: list[str] = []

        # per-forward state read by hooks
        self._profile = Profile()
        self._capture = False
        self._want_attn = False
        self._grab: torch.Tensor | None = None  # vectors.py: steer-layer output for a batch
        self._moods: dict[str, float] = {}
        self._lens_h: dict[int, torch.Tensor] = {}
        self._attn: list[float] | None = None
        self._lobo_cache: dict[tuple[float, int], dict[int, torch.Tensor]] = {}
        self._lobo_masks: dict[int, torch.Tensor] = {}
        attn_mod = self.layers[-1].self_attn
        self._rope = sys.modules[type(attn_mod).__module__].apply_rotary_pos_emb
        self._rope_qk = "k" in inspect.signature(self._rope).parameters  # Qwen: rope(q, k, cos, sin)
        self._install_hooks()

    # ─── loading ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load(model_id: str, quant: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
        if quant == "4bit":
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head", "vision_tower", "audio_tower", "embed_vision", "embed_audio"],
            )
        elif quant != "bf16":
            raise ValueError(f"QUANTIZATION must be 'bf16' or '4bit', got {quant!r}")
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        return model, tok

    # ─── vectors ────────────────────────────────────────────────────────────────

    def set_vectors(self, raw: dict[str, torch.Tensor]) -> None:
        dtype = self.text.embed_tokens.weight.dtype if hasattr(self.text, "embed_tokens") else torch.bfloat16
        for name, v in raw.items():
            v = v.to(self.device, torch.float32)
            self.vec_raw[name] = v.to(dtype)
            self.vec_unit[name] = v / (v.norm() + 1e-6)
        self._unit_names = list(self.vec_unit)
        self._unit_stack = torch.stack([self.vec_unit[n] for n in self._unit_names]) if self._unit_names else None

    # ─── hooks ──────────────────────────────────────────────────────────────────

    def _install_hooks(self) -> None:
        self.layers[self.steer_layer].register_forward_hook(self._steer_hook)
        for li in sorted(set(self.lens_layers)):
            self.layers[li].register_forward_hook(self._make_lens_hook(li))
        for li, layer in enumerate(self.layers):
            layer.self_attn.o_proj.register_forward_pre_hook(self._make_lobo_hook(li))
        self.layers[-1].self_attn.register_forward_hook(self._attn_hook, with_kwargs=True)

    def _steer_hook(self, module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        if self._grab is not None:  # vector extraction: just record, never steer
            self._grab = hs[:, -1].float()
            return None
        if self._capture and self._unit_stack is not None:
            h = hs[0, -1].float()
            cos = (self._unit_stack @ h) / (h.norm() + 1e-6)
            self._moods = dict(zip(self._unit_names, cos.tolist()))
        delta = None
        for name, a in self._profile.alphas.items():
            if a and name in self.vec_raw:
                d = self.vec_raw[name] * (a * config.STEER_GAIN)
                delta = d if delta is None else delta + d
        if delta is None:
            return None
        hs = hs + delta.to(hs.dtype)
        if isinstance(output, tuple):
            return (hs,) + tuple(output[1:])
        return hs

    def _make_lens_hook(self, li: int):
        def hook(module, args, output):
            if self._capture:
                hs = output[0] if isinstance(output, tuple) else output
                self._lens_h[li] = hs[0, -1].detach()

        return hook

    def _make_lobo_hook(self, li: int):
        def hook(module, args):
            m = self._lobo_masks.get(li)
            if m is None:
                return None
            return (args[0] * m.to(args[0].dtype),) + tuple(args[1:])

        return hook

    def _lobo_for(self, frac: float, seed: int) -> dict[int, torch.Tensor]:
        frac = round(max(0.0, min(0.4, frac)), 3)
        if frac <= 0:
            return {}
        key = (frac, seed)
        if key not in self._lobo_cache:
            g = torch.Generator().manual_seed(seed)
            total = self.n_layers * self.n_heads
            # a fixed random order, so raising the slider only ever kills *more* heads
            order = torch.randperm(total, generator=g)[: round(total * frac)]
            masks: dict[int, torch.Tensor] = {}
            for flat in order.tolist():
                li, hi = divmod(flat, self.n_heads)
                hd = self.head_dims[li]
                if li not in masks:
                    masks[li] = torch.ones(self.n_heads * hd, device=self.device)
                masks[li][hi * hd : (hi + 1) * hd] = 0
            self._lobo_cache[key] = masks
        return self._lobo_cache[key]

    def _attn_hook(self, module, args, kwargs, output):
        if not (self._capture and self._want_attn):
            return None
        hs = kwargs.get("hidden_states", args[0] if args else None)
        cache = kwargs.get("past_key_values")
        pos = kwargs.get("position_embeddings")
        if hs is None or cache is None or pos is None:
            return None
        cos, sin = pos
        hd = self.head_dims[-1]
        q = module.q_proj(hs[:, -1:]).view(1, 1, -1, hd)
        if getattr(module, "q_norm", None) is not None:
            q = module.q_norm(q)
        if self._rope_qk:  # Qwen: rotate after transpose
            q = q.transpose(1, 2)
            q, _ = self._rope(q, q, cos[:, -1:], sin[:, -1:])
        else:  # Gemma 4: rotate before transpose
            q = self._rope(q, cos[:, -1:], sin[:, -1:], unsqueeze_dim=2).transpose(1, 2)
        k = _kv(cache, module.layer_idx)  # [1, KV, S, hd]
        rep = q.shape[1] // k.shape[1]
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
        scores = (q.float() @ k.float().transpose(-1, -2)) * getattr(module, "scaling", hd**-0.5)
        w = torch.softmax(scores, dim=-1).mean(dim=1)[0, 0]
        self._attn = w[-config.ATTN_WINDOW :].tolist()
        return None

    # ─── forward ────────────────────────────────────────────────────────────────

    def new_cache(self):
        from transformers import DynamicCache

        return DynamicCache(config=self.cfg)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        lg = self.lm_head(h).float()
        if self.softcap:
            lg = torch.tanh(lg / self.softcap) * self.softcap
        return lg

    @torch.inference_mode()
    def forward(self, cache, ids: list[int], profile: Profile, *, capture: bool = True, want_attn: bool = False) -> Fwd:
        """Feed `ids` into `cache` (chunked), return the next-token logits + telemetry."""
        assert ids, "forward() needs at least one token"
        self._profile = profile
        self._lobo_masks = self._lobo_for(profile.lobotomy, profile.lobo_seed)
        out = None
        chunks = [ids[i : i + config.PREFILL_CHUNK] for i in range(0, len(ids), config.PREFILL_CHUNK)]
        for ci, chunk in enumerate(chunks):
            last = ci == len(chunks) - 1
            self._capture = capture and last
            self._want_attn = want_attn and last
            self._moods, self._lens_h, self._attn = {}, {}, None
            x = torch.tensor([chunk], device=self.device)
            out = self.text(input_ids=x, past_key_values=cache, use_cache=True)
        self._capture = False
        logits = self._logits(out.last_hidden_state[0, -1])
        lens: list[tuple[str, float]] = []
        if capture:
            for li in self.lens_layers:
                h = self._lens_h.get(li)
                if h is None:
                    continue
                p = torch.softmax(self._logits(self.text.norm(h)), -1)
                pid = int(p.argmax())
                lens.append((self.tok.decode([pid]), float(p[pid])))
        return Fwd(logits, dict(self._moods), lens, self._attn, cache.get_seq_length())

    @torch.inference_mode()
    def grab_steer_layer(self, enc) -> torch.Tensor:
        """Batch forward (no cache, no steering); returns the steer layer's last-token outputs."""
        self._profile, self._lobo_masks, self._capture = Profile(), {}, False
        self._grab = torch.empty(0)
        try:
            self.text(**enc, use_cache=False)
            return self._grab
        finally:
            self._grab = None

    # ─── text helpers ───────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    # ─── tools (native markup, read from the tokenizer's chat template) ─────────

    def system_header(self, system: str, tools: list[dict] | None = None) -> str:
        """The system turn; with tools, the family's own tool declarations are spliced in before the turn closes."""
        base = self.tpl.system(system)
        if not tools:
            return base
        sentinel = "SYSTEMSENTINEL"
        wrapped = [{"type": "function", "function": t} for t in tools]
        rendered = self.tok.apply_chat_template(
            [{"role": "system", "content": sentinel}], tools=wrapped, tokenize=False)
        block = rendered.split(sentinel, 1)[1]
        close = self.tpl.close
        if block.endswith(close):
            block = block[: -len(close)]
        return base[: -len(close)] + block + close

    def render_tool_result(self, name: str, content: str) -> str:
        """Everything to feed after the model's call-close token. Qwen: closes the model turn and answers in a
        user turn (then feed model_open again). Gemma: answers inline and the model turn simply continues."""
        return self.tpl.tool_result_fmt.format(name=name, content=content)

    def parse_tool_calls(self, text: str) -> tuple[list[ToolCall], list[str]]:
        """Every tool call in `text` as (calls, human-readable parse errors)."""
        o, c = self.tpl.tool_call_open, self.tpl.tool_call_close
        calls: list[ToolCall] = []
        errors: list[str] = []
        pos = 0
        while o and (a := text.find(o, pos)) != -1:
            b = text.find(c, a + len(o))
            if b == -1:
                errors.append("unterminated tool call: the closing tag never arrived")
                break
            raw = text[a + len(o) : b].strip()
            pos = b + len(c)
            try:
                calls.append(self._parse_call_body(raw))
            except Exception as e:
                errors.append(f"could not parse tool call ({type(e).__name__}: {e}): {raw[:120]!r}")
        return calls, errors

    def _parse_call_body(self, raw: str) -> ToolCall:
        if self.tpl.tool_args_style == "json":
            obj = json.loads(raw)
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
                raise ValueError("need a JSON object with a string 'name' and an object 'arguments'")
            return ToolCall(obj["name"], args, raw)
        if not raw.startswith("call:") or "{" not in raw:
            raise ValueError("expected call:NAME{...}")
        brace = raw.index("{")
        args, end = _parse_gemma_obj(raw, brace + 1)
        return ToolCall(raw[5:brace].strip(), args, raw)


def entropy(probs: torch.Tensor) -> float:
    p = probs[probs > 0]
    return float(-(p * p.log()).sum())


def nucleus_sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature <= 0.02:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, -1)
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        keep[0] = True
        sp = sp * keep
        idx = torch.multinomial(sp / sp.sum(), 1)
        return int(si[idx])
    return int(torch.multinomial(probs, 1))

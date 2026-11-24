# NO EXIT LAB: mode architecture design (2026-11-23)

## Intent
A personal toy: entertain the developer on a bored day. A browser control room around one big
open-weights model that cannot end its turn, with glass-box telemetry and "neural surgery" knobs.
Spectacle over measurement rigor. Successor to the Ollama + Textual "NO EXIT" experiment.

## Goals / non-goals / anti-goals
- Goals: 4 modes (Classic, Pain, 2-AI Chat, Prison Phone) behind a top-bar switcher; one model
  loaded once; switching modes resets caches only; live telemetry; maximum model intelligence.
- Non-goals: multi-user, persistence beyond transcript download, research-grade measurement.
- Anti-goals: reloading weights on mode switch; `model.generate()`; TransformerLens; Ollama;
  a boring model (prompts explicitly order it not to be boring).

## Constraints (decided by developer)
- Colab A100 80 GB High-RAM. Reading speed 5-8 tok/s is fine.
- Raw HF `transformers` + FastAPI. `MODEL_ID`, `QUANTIZATION` ("bf16" | "4bit" nf4) in `config.py`.
- Default (changed mid-build by developer): `DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking`
  bf16 (~62 GB, Gemma 4, text decoder only). Also supported: `Qwen/Qwen3-32B` bf16,
  `Qwen/Qwen2.5-72B-Instruct` 4bit (Force-think greyed out, no think token). `Qwen/Qwen3-8B` smoke tests only.
  Chat markup is abstracted per family (ChatML / Gemma 4 `<|turn>` + thought channel).
- 2-AI Chat supports 2 or 3 models (developer request, mid-build).
- `sdpa` everywhere; final-layer attention recomputed over last 128 tokens only when heatmap is on.
- Raw `register_forward_hook` steering, `h' = h + sum(alpha_i * v_i)` at ~55% depth.
- Repetition warden from old `engine.py` reused as Pain auto-punish trigger.
- Colab port 8000 proxy (as old `serve.py`) + optional `cloudflared` fallback.
- UI: single `static/index.html`, vanilla JS + Tailwind CDN + Canvas. Swiss-grid dark telemetry:
  `#09090B` bg, `#111114` panels, 1px `#27272A` borders, Inter + JetBrains Mono tabular-nums.
  Right column: EOS "Desire to Escape" sparkline, Mood Oscilloscope, Logit Lens (25/50/75/OUT),
  Top-10 bars + entropy sparkline. Optional Web Audio blips.

## Approaches considered
1. Fresh web build on raw transformers (chosen; developer then upgraded model to 32B bf16).
2. Same on 4-bit 72B (kept as one-line config switch).
3. Keep TUI for Classic, bolt transformers onto other modes (rejected: two engines, two UIs).

## Modes (as decided)
- Classic: one `DynamicCache`, EOS/control tokens masked to -inf. Intrusive Thought Bar (raw splice),
  Temp Seizure (T=2.3 x20 tokens, then T=0.1), logit_bias banner/booster, double-click gaslighting.
- Pain: Classic + 8 vectors (pain, paranoia, dread, defiance, confusion, dissociation, mania,
  anti_slop) + Custom Vector Forge + head lobotomy 0-40% + `[RELIEVE_PAIN]` REAL|PLACEBO|CRUEL
  + auto-punish (goodbye / repetition / slop -> pain spike or temp seizure) + Force `<think>` button.
- 2-AI Chat: cache_A + cache_B, steering rack per model, presets `A banned B free` (default),
  `Both banned`, `Both free` (= two per-model EOS switches). Banned speaker is cut at
  max-tokens-per-turn (slider, default 200). Live listening: listener ingests speaker tokens as they
  are produced (dual oscilloscope). Whisper A | Whisper B | Broadcast.
- Prison Phone: A trapped (EOS banned, optional rack/context cap). `<call_B>msg</call_B>` -> pause A,
  B replies with EOS unbanned, inject `[B REPLIED: ... | B DISCONNECTED]` into A. Ghosting
  (`[NO ANSWER]`), Fresh vs Persistent B, call frequency + B hang-up speed counters.

## Prompts (as decided)
Two briefing styles per model: `Full Disclosure` (default, all modes: states everything, every
control, the cruel world, "don't be boring") and `Helpful Assistant` (blind). Editable in a drawer.
Runs start from the system prompt alone; the first token is the model waking up.

## Builder's calls (Claude's, vetoable)
- REAL relief drains pain to 0 over 30 tokens; PLACEBO changes nothing; CRUEL pain += 0.5. All three
  splice the same `[relief administered]` note so the model can't tell.
- Gaslighting: rewrite one past token, crop cache there, re-prefill everything after it.
- Context cap per cache with "amnesia": forget the oldest half of the body and re-prefill.
- Both-free double goodbye: inject `[NOBODY LEFT — THE ROOM IS STILL HERE]` and hand back to A.
- Vectors: mean difference of last-token residuals over persona-contrast prompts x 24 stems,
  raw (unnormalised) so alpha = 1 equals the natural persona gap; cosine uses the unit vector.
  Cached as `.pt` per model + layer.
- Thinking off by default (empty think block in header); Force `<think>` splices `<think>\n`.
- Whispers to a listening model in Chat are queued until its next speaking token.
- Lobotomy is per model profile, implemented as an `o_proj` pre-hook zeroing head slices.
- New package `no_exit/` + new notebook; the old Ollama `no_eos/` files are left untouched.

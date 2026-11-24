"""python -m no_eos [--model qwen3.8:27b] [--prompt "..."] [--fake] [--probe 40]"""

from __future__ import annotations

import argparse
import os
import time

from .engine import Engine, Settings


def main() -> None:
    p = argparse.ArgumentParser(prog="no_eos", description="A model that cannot end its turn.")
    p.add_argument("--model", default=os.environ.get("NO_EOS_MODEL", "qwen3.8:27b"))
    p.add_argument("--host", default=os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434"))
    p.add_argument("--prompt", default=None, help="skip the intro screen")
    p.add_argument("--system", default="", help="optional system prompt")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--rep", type=float, default=1.5, help="repetition punishment strength")
    p.add_argument("--ctx", type=int, default=32768, help="context window before amnesia kicks in")
    p.add_argument("--no-think", action="store_true", help="skip the thinking phase")
    p.add_argument("--fake", action="store_true", help="demo with a fake model (no GPU needed)")
    p.add_argument("--probe", type=int, default=0, metavar="N", help="no TUI: run N tokens and print a report")
    a = p.parse_args()

    settings = Settings(temperature=a.temp, rep_strength=a.rep, num_ctx=a.ctx, think=not a.no_think)
    if a.fake:
        from .backends import FakeBackend

        backend = FakeBackend()
    else:
        from .backends import OllamaBackend

        backend = OllamaBackend(a.model, a.host, num_ctx=a.ctx)

    if a.probe:
        probe(backend, a.prompt or "Write a haiku about rain.", a.system, settings, a.probe)
        return

    from .app import NoExitApp

    NoExitApp(backend, a.prompt, a.system, settings).run()


def probe(backend, prompt: str, system: str, settings: Settings, n: int) -> None:
    """Headless sanity check: proves logprobs work and shows the speed."""
    eng = Engine(backend, prompt, system, settings)
    t0 = time.perf_counter()
    for _ in range(n):
        st = eng.step()
        mark = " ⊘" if st.escape else (" ↻" if st.rep_dodge else "")
        print(f"{st.index:4d}  {st.chosen.prob:5.0%}  {st.chosen.text!r:<18} "
              f"urge={st.stop_prob:4.0%}  cands={len(st.candidates)}{mark}", flush=True)
    dt = time.perf_counter() - t0
    print("\n--- report ---")
    print(f"{n} tokens in {dt:.1f}s  →  {n / dt:.1f} tok/s")
    print(f"escape attempts: {eng.escapes}   sampler-forced exits: {eng.hard_escapes}   repeat dodges: {eng.rep_dodges}")
    print("text:\n" + "".join(eng.tokens))


if __name__ == "__main__":
    main()

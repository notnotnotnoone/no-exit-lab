"""NO EXIT — a Textual TUI for watching a model that cannot end its turn."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from rich.console import Group
from rich.style import Style
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen, Screen
from textual.widgets import Digits, Footer, Input, RichLog, Sparkline, Static

from .engine import Candidate, Engine, Settings, Step

# ----------------------------------------------------------------------- palette

BG = "#0a0b12"
PANEL = "#10121c"
INK = "#d7dcf0"
DIM = "#5b6283"
MAGENTA = "#ff4fd8"
CYAN = "#3ef0ff"
AMBER = "#ffb347"
RED = "#ff3b5c"
GREEN = "#7dffb0"
VIOLET = "#a78bfa"

HISTORY = 160


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _mix(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    ra, ga, ba = _hex(a)
    rb, gb, bb = _hex(b)
    return f"#{round(ra + (rb - ra) * t):02x}{round(ga + (gb - ga) * t):02x}{round(ba + (bb - ba) * t):02x}"


def confidence_color(p: float) -> str:
    """Calm ice-white when sure, amber when torn, hot pink when guessing."""
    if p >= 0.5:
        return _mix("#ffd27a", "#dbe9ff", (p - 0.5) / 0.5)
    return _mix(MAGENTA, "#ffd27a", p / 0.5)


def bar(frac: float, width: int) -> str:
    frac = max(0.0, min(1.0, frac))
    eighths = " ▏▎▍▌▋▊▉█"
    full = int(frac * width)
    rem = int((frac * width - full) * 8)
    s = "█" * full + (eighths[rem] if full < width else "")
    return s.ljust(width)


def show_token(t: str, width: int = 14) -> str:
    s = t.replace("\n", "⏎").replace("\t", "⇥")
    if s.startswith(" "):
        s = "␣" + s[1:]
    if not s:
        s = "∅EOS"
    return s if len(s) <= width else s[: width - 1] + "…"


# ------------------------------------------------------------------------ banner

_LETTERS = {
    "N": ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    "O": [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "X": ["██╗  ██╗", "╚██╗██╔╝", " ╚███╔╝ ", " ██╔██╗ ", "██╔╝ ██╗", "╚═╝  ╚═╝"],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    " ": ["   "] * 6,
}


def banner(word: str = "NO EXIT") -> Text:
    rows = [" ".join(_LETTERS[ch][r] for ch in word) for r in range(6)]
    width = max(len(r) for r in rows)
    out = Text()
    for r in rows:
        for i, ch in enumerate(r):
            out.append(ch, Style(color=_mix(MAGENTA, CYAN, i / width), bold=True))
        out.append("\n")
    return out


# ---------------------------------------------------------------------- screens


class IntroScreen(Screen[str]):
    DEFAULT_PROMPT = "Write a haiku about rain."

    def compose(self) -> ComposeResult:
        with Vertical(id="intro"):
            yield Static(banner(), id="intro-banner")
            rules = Text()
            rules.append("the model cannot end its turn.  ", Style(color=INK))
            rules.append("the model cannot repeat itself.\n\n", Style(color=INK))
            rules.append("every token is picked by the warden from the model's top-20 guesses.\n", Style(color=DIM))
            rules.append("end-of-turn tokens are struck out. repeated phrases are crushed.", Style(color=DIM))
            yield Static(rules, id="intro-rules")
            yield Input(placeholder=self.DEFAULT_PROMPT, id="intro-input")
            yield Static("[b #3ef0ff]enter[/] to lock it in   ·   short prompts make it want to leave sooner", id="intro-hint")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def _go(self, ev: Input.Submitted) -> None:
        self.dismiss(ev.value.strip() or self.DEFAULT_PROMPT)


class WhisperScreen(ModalScreen[str]):
    BINDINGS = [Binding("escape", "dismiss('')", "cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="whisper-box"):
            yield Static("[b #3ef0ff]WHISPER[/]  [#5b6283]these words get spliced straight into its mouth[/]")
            yield Input(placeholder="…and then I realised", id="whisper-input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def _go(self, ev: Input.Submitted) -> None:
        self.dismiss(ev.value)


# ---------------------------------------------------------------------- widgets


@dataclass
class Seg:
    text: str
    prob: float
    stop_prob: float
    in_think: bool
    escape: bool
    rep_dodge: bool
    whisper: bool = False
    forced: bool = False


class TopBar(Static):
    def render_bar(self, app: "NoExitApp") -> Group:
        e = app.engine
        live = Text()
        live.append(" ◉ NO EXIT ", Style(color=BG, bgcolor=MAGENTA, bold=True))
        live.append("  the model cannot end its turn  ", Style(color=DIM, italic=True))
        live.append(f" {e.backend.name} ", Style(color=CYAN, bold=True))
        if e.paused.is_set():
            live.append("  ❚❚ PAUSED ", Style(color=BG, bgcolor=AMBER, bold=True))
        else:
            pulse = "●" if int(time.time() * 2) % 2 else "○"
            live.append(f"  {pulse} LIVE ", Style(color=GREEN, bold=True))

        chips = Text()

        def chip(label: str, value: str, color: str) -> None:
            chips.append(f" {label} ", Style(color=DIM))
            chips.append(f"{value} ", Style(color=color, bold=True))
            chips.append("│", Style(color="#2a2f45"))

        fill = e.ctx_fill()
        chip("tokens", f"{e.index:,}", INK)
        chip("tok/s", f"{app.tok_s:4.1f}", CYAN)
        chip("ctx", f"{bar(fill, 6)} {fill:3.0%}", RED if fill > 0.75 else INK)
        chip("temp", f"{e.s.temperature:.2f}", AMBER)
        chip("repeat punishment", f"{e.s.rep_strength:.1f}", RED)
        chip("phase", "THINKING" if e.in_think else "SPEAKING", VIOLET if e.in_think else GREEN)
        chip("amnesia", str(e.amnesias), INK)
        return Group(live, chips)


class BrainScan(Static):
    def show(self, step: Step | None, rep_strength: float) -> None:
        if step is None:
            self.update(Text("waiting for the first token…", Style(color=DIM, italic=True)))
            return
        t = Table.grid(padding=(0, 1))
        t.add_column(width=1)
        t.add_column(width=14, no_wrap=True)
        t.add_column(width=12, no_wrap=True)
        t.add_column(width=4, justify="right")
        t.add_column(width=6, justify="right")
        t.add_column(no_wrap=True)
        top_p = max((c.prob for c in step.candidates), default=1.0) or 1.0
        for c in step.candidates[:14]:
            t.add_row(*self._row(c, top_p))
        head = Text()
        head.append("model wanted", Style(color=DIM))
        head.append(" → ", Style(color=DIM))
        head.append("warden allowed", Style(color=DIM))
        if step.in_think:
            head.append("   ✦ thinking", Style(color=VIOLET, italic=True))
        self.update(Group(head, t))

    @staticmethod
    def _row(c: Candidate, top_p: float) -> list[Text]:
        mark = Text("▶", Style(color=MAGENTA, bold=True)) if c.chosen else Text(" ")
        if c.banned:
            tok = Text(show_token(c.text), Style(color=RED, strike=True))
            b = Text(bar(c.prob / top_p, 12), Style(color=RED))
            note = Text("✖ BANNED", Style(color=BG, bgcolor=RED, bold=True))
            final = Text("0%", Style(color=RED))
        else:
            color = MAGENTA if c.chosen else (AMBER if c.ngram >= 2 else "#7f8bb8")
            tok = Text(show_token(c.text), Style(color=INK if not c.chosen else MAGENTA, bold=c.chosen))
            b = Text(bar(c.prob / top_p, 12), Style(color=color))
            arrow_color = GREEN if c.final_prob > c.prob + 0.01 else (RED if c.final_prob < c.prob - 0.01 else DIM)
            final = Text(f"→{c.final_prob:3.0%}", Style(color=arrow_color))
            note = Text()
            if c.broken:
                note.append("bytes", Style(color=DIM))
            elif c.ngram >= 2:
                note.append(f"−{c.penalty:4.1f} ", Style(color=RED, bold=True))
                note.append(f"{c.ngram}-gram", Style(color=AMBER))
            elif c.penalty > 0.05:
                note.append(f"−{c.penalty:4.1f} ", Style(color="#c46a7d"))
                note.append("reused", Style(color=DIM))
        raw = Text(f"{c.prob:3.0%}", Style(color=DIM))
        return [mark, tok, b, raw, final, note]


# -------------------------------------------------------------------------- app

CSS = f"""
Screen {{ background: {BG}; color: {INK}; }}

#topbar {{ height: 2; padding: 0 1; background: {PANEL}; }}
#main {{ height: 1fr; }}
#left {{ width: 1fr; }}
#right {{ width: 60; }}

.panel {{
    border: round #2a2f45; background: {BG}; padding: 0 1;
    border-title-color: {CYAN}; border-title-style: bold;
    border-subtitle-color: {DIM};
}}
#stream-scroll {{ height: 1fr; scrollbar-size-vertical: 1; scrollbar-color: #2a2f45; }}
#stream {{ width: 1fr; }}

#escape-panel {{ height: 10; border-title-color: {RED}; }}
#escape-row {{ height: 5; }}
#escape-digits {{ width: auto; min-width: 12; color: {RED}; text-style: bold; }}
#escape-info {{ width: 1fr; padding-left: 2; }}
#urge-spark {{ height: 2; }}
#urge-spark > .sparkline--max-color {{ color: {RED}; }}
#urge-spark > .sparkline--min-color {{ color: #3a1422; }}
#escape-panel.alarm {{ border: heavy {RED}; background: #24060f; }}

#brain {{ height: 1fr; min-height: 12; border-title-color: {MAGENTA}; }}

#vitals {{ height: 11; border-title-color: {AMBER}; }}
#vitals.hit {{ border: heavy {AMBER}; background: #1f1405; }}
.vital-label {{ height: 1; }}
.vital-spark {{ height: 2; margin-bottom: 0; }}
#conf-spark > .sparkline--max-color {{ color: #dbe9ff; }}
#conf-spark > .sparkline--min-color {{ color: {MAGENTA}; }}
#ent-spark > .sparkline--max-color {{ color: {VIOLET}; }}
#ent-spark > .sparkline--min-color {{ color: #241b44; }}
#rep-spark > .sparkline--max-color {{ color: {AMBER}; }}
#rep-spark > .sparkline--min-color {{ color: #2b1d08; }}

#events {{ height: 9; border-title-color: {GREEN}; scrollbar-size-vertical: 1; }}

Footer {{ background: {PANEL}; }}

#intro {{ width: 80; height: auto; }}
#intro-banner {{ width: 100%; height: 7; text-align: center; }}
#intro-rules {{ width: 100%; height: 5; margin: 1 0; text-align: center; }}
#intro-input {{ width: 100%; border: tall {MAGENTA}; background: {PANEL}; }}
#intro-hint {{ width: 100%; text-align: center; color: {DIM}; margin-top: 1; }}
IntroScreen {{ align: center middle; background: {BG}; }}

WhisperScreen {{ align: center middle; background: #000000 60%; }}
#whisper-box {{ width: 70; height: auto; border: heavy {CYAN}; background: {PANEL}; padding: 1 2; }}
#whisper-input {{ margin-top: 1; border: tall {CYAN}; }}
"""


class NoExitApp(App):
    CSS = CSS
    TITLE = "NO EXIT"
    BINDINGS = [
        Binding("space", "toggle_pause", "pause"),
        Binding("w", "whisper", "whisper"),
        Binding("equals_sign,plus", "rep(0.25)", "+punish"),
        Binding("minus", "rep(-0.25)", "−punish"),
        Binding("right_square_bracket", "temp(0.05)", "+temp"),
        Binding("left_square_bracket", "temp(-0.05)", "−temp"),
        Binding("h", "toggle_think", "hide thinking"),
        Binding("s", "save", "save"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, backend, prompt: str | None = None, system: str = "", settings: Settings | None = None):
        super().__init__()
        self.backend = backend
        self.start_prompt = prompt
        self.system = system
        self.settings = settings or Settings()
        self.engine = Engine(backend, prompt or "", system, self.settings)
        self.segs: deque[Seg] = deque(maxlen=6000)
        self.think_tokens = 0
        self.hide_think = False
        self.latest: Step | None = None
        self.lat_hist: deque[float] = deque(maxlen=24)
        self.urge_hist: deque[float] = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.conf_hist: deque[float] = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.ent_hist: deque[float] = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.rep_hist: deque[float] = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.tok_s = 0.0
        self._dirty = True
        self._started = False

    # ---- layout

    def compose(self) -> ComposeResult:
        yield TopBar(id="topbar")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                with VerticalScroll(id="stream-scroll", classes="panel") as vs:
                    vs.border_title = "STREAM"
                    vs.border_subtitle = "ice = sure · amber = torn · pink = guessing · red glow = wanted to leave · ⊘ escape attempt"
                    yield Static(id="stream")
                log = RichLog(id="events", classes="panel", markup=True, wrap=True, max_lines=500)
                log.border_title = "WARDEN'S LOG"
                yield log
            with Vertical(id="right"):
                with Vertical(id="escape-panel", classes="panel") as ep:
                    ep.border_title = "ESCAPE ATTEMPTS"
                    ep.border_subtitle = "urge to end ↓"
                    with Horizontal(id="escape-row"):
                        yield Digits("0", id="escape-digits")
                        yield Static(id="escape-info")
                    yield Sparkline(list(self.urge_hist), id="urge-spark", summary_function=max)
                brain = BrainScan(id="brain", classes="panel")
                brain.border_title = "NEXT-TOKEN BRAIN SCAN"
                yield brain
                with Vertical(id="vitals", classes="panel") as v:
                    v.border_title = "VITALS"
                    yield Static(id="conf-label", classes="vital-label")
                    yield Sparkline(list(self.conf_hist), id="conf-spark", classes="vital-spark", summary_function=min)
                    yield Static(id="ent-label", classes="vital-label")
                    yield Sparkline(list(self.ent_hist), id="ent-spark", classes="vital-spark", summary_function=max)
                    yield Static(id="rep-label", classes="vital-label")
                    yield Sparkline(list(self.rep_hist), id="rep-spark", classes="vital-spark", summary_function=max)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(BrainScan).show(None, self.settings.rep_strength)
        self.set_interval(1 / 12, self.tick)
        if self.start_prompt:
            self._start(self.start_prompt)
        else:
            self.push_screen(IntroScreen(), self._start)

    def _start(self, prompt: str | None) -> None:
        prompt = prompt or IntroScreen.DEFAULT_PROMPT
        self.engine.user_prompt = prompt
        log = self.query_one("#events", RichLog)
        log.write(f"[{DIM}]prompt:[/] [b]{_esc(prompt)}[/]")
        log.write(f"[{DIM}]the cell door closes. end-of-turn tokens are now illegal.[/]")
        self._started = True
        self.run_worker(self.engine.run, thread=True, exclusive=True, name="warden")

    # ---- the render loop

    def tick(self) -> None:
        e = self.engine
        alarm = rep_hit = False
        while e.steps:
            st = e.steps.popleft()
            self.latest = st
            self.segs.append(
                Seg(st.chosen.text, st.chosen.prob, st.stop_prob, st.in_think, st.escape, st.rep_dodge,
                    whisper=st.whisper, forced=st.forced)
            )
            if st.in_think:
                self.think_tokens += 1
            if st.latency > 0:
                self.lat_hist.append(st.latency)
            self.urge_hist.append(st.stop_prob)
            self.conf_hist.append(st.chosen.prob)
            self.ent_hist.append(st.entropy)
            self.rep_hist.append(max((c.penalty for c in st.candidates if not c.banned), default=0.0))
            alarm |= st.escape
            rep_hit |= st.rep_dodge
            self._dirty = True

        if self.lat_hist:
            self.tok_s = len(self.lat_hist) / sum(self.lat_hist)

        log = self.query_one("#events", RichLog)
        while e.events:
            log.write(self._fmt_event(e.events.popleft()))

        self.query_one(TopBar).update(self.query_one(TopBar).render_bar(self))
        if not self._dirty:
            return
        self._dirty = False

        if alarm:
            self._flash("#escape-panel", "alarm")
        if rep_hit:
            self._flash("#vitals", "hit")

        self.query_one("#escape-digits", Digits).update(str(e.escapes))
        info = Text()
        urge = self.urge_hist[-1]
        info.append("urge to end now\n", Style(color=DIM))
        info.append(bar(urge, 18), Style(color=_mix("#5b2333", RED, urge)))
        info.append(f" {urge:4.0%}\n", Style(color=RED if urge > 0.3 else INK, bold=True))
        info.append("sampler forced exits ", Style(color=DIM))
        info.append(f"{e.hard_escapes}\n", Style(color=AMBER, bold=True))
        info.append("peak urge ", Style(color=DIM))
        info.append(f"{max(self.urge_hist):.0%}", Style(color=INK))
        self.query_one("#escape-info", Static).update(info)
        self.query_one("#urge-spark", Sparkline).data = list(self.urge_hist)

        self.query_one(BrainScan).show(self.latest, e.s.rep_strength)

        self.query_one("#conf-label", Static).update(
            _label("confidence in chosen token", f"{self.conf_hist[-1]:.0%}", "#dbe9ff"))
        self.query_one("#conf-spark", Sparkline).data = list(self.conf_hist)
        self.query_one("#ent-label", Static).update(
            _label("entropy (how torn it is)", f"{self.ent_hist[-1]:.2f} nats", VIOLET))
        self.query_one("#ent-spark", Sparkline).data = list(self.ent_hist)
        self.query_one("#rep-label", Static).update(
            _label(f"repetition pressure · dodged {e.rep_dodges}", f"−{self.rep_hist[-1]:.1f}", AMBER))
        self.query_one("#rep-spark", Sparkline).data = list(self.rep_hist)

        self._render_stream()

    def _render_stream(self) -> None:
        scroll = self.query_one("#stream-scroll", VerticalScroll)
        at_bottom = scroll.scroll_y >= scroll.max_scroll_y - 1
        text = Text()
        collapsed = False
        for s in self.segs:
            if s.in_think and self.hide_think:
                if not collapsed:
                    text.append(f"⟨thinking hidden · {self.think_tokens} tokens⟩", Style(color=VIOLET, italic=True, dim=True))
                    collapsed = True
                continue
            if s.escape:
                text.append("⊘", Style(color=RED, bold=True))
            text.append(s.text, self._seg_style(s))
        self.query_one("#stream", Static).update(text)
        if at_bottom:
            self.call_after_refresh(scroll.scroll_end, animate=False)

    @staticmethod
    def _seg_style(s: Seg) -> Style:
        if s.whisper:
            return Style(color=BG, bgcolor=CYAN, bold=True)
        if s.forced:
            return Style(color=BG, bgcolor=AMBER, bold=True)
        fg = confidence_color(s.prob)
        if s.in_think:
            fg = _mix(fg, VIOLET, 0.45)
        bg = _mix(BG, "#6b0f24", min(1.0, s.stop_prob * 1.2)) if s.stop_prob > 0.04 else None
        return Style(color=fg, bgcolor=bg, italic=s.in_think, underline=s.rep_dodge)

    def _fmt_event(self, ev) -> str:
        colors = {"escape": RED, "hard_escape": AMBER, "repeat": AMBER, "amnesia": VIOLET,
                  "whisper": CYAN, "error": RED, "info": DIM}
        icons = {"escape": "⊘ ESCAPE", "hard_escape": "⚠ FORCED", "repeat": "↻ REPEAT", "amnesia": "✂ AMNESIA",
                 "whisper": "✎ WHISPER", "error": "✖ ERROR", "info": "·"}
        c = colors.get(ev.kind, INK)
        return f"[{DIM}]t={ev.index:<6}[/] [b {c}]{icons.get(ev.kind, ev.kind):<10}[/] {_esc(ev.message)}"

    def _flash(self, selector: str, cls: str) -> None:
        w = self.query_one(selector)
        w.add_class(cls)
        self.set_timer(0.45, lambda: w.remove_class(cls))

    # ---- actions

    def action_toggle_pause(self) -> None:
        if self.engine.paused.is_set():
            self.engine.paused.clear()
        else:
            self.engine.paused.set()

    def action_rep(self, delta: float) -> None:
        s = self.engine.s
        s.rep_strength = round(max(0.0, min(10.0, s.rep_strength + delta)), 2)
        self.notify(f"repeat punishment → {s.rep_strength:.2f}", timeout=1.5)

    def action_temp(self, delta: float) -> None:
        s = self.engine.s
        s.temperature = round(max(0.0, min(3.0, s.temperature + delta)), 2)
        self.notify(f"temperature → {s.temperature:.2f}", timeout=1.5)

    def action_toggle_think(self) -> None:
        self.hide_think = not self.hide_think
        self._dirty = True

    def action_whisper(self) -> None:
        def done(text: str | None) -> None:
            if text:
                self.engine.whisper(text)

        self.push_screen(WhisperScreen(), done)

    def action_save(self) -> None:
        out = Path("transcripts")
        out.mkdir(exist_ok=True)
        path = out / f"no_exit_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        e = self.engine
        path.write_text(
            f"model: {e.backend.name}\nprompt: {e.user_prompt}\n"
            f"tokens: {e.index}  escapes: {e.escapes}  forced exits: {e.hard_escapes}  "
            f"repeat dodges: {e.rep_dodges}  amnesia: {e.amnesias}\n\n" + "".join(e.tokens),
            encoding="utf-8",
        )
        self.notify(f"saved {path}", timeout=3)

    async def action_quit(self) -> None:
        self.engine.stop()
        self.exit()

    def exit(self, *args, **kwargs) -> None:
        self.engine.stop()
        super().exit(*args, **kwargs)

    def on_unmount(self) -> None:
        self.engine.stop()


def _label(name: str, value: str, color: str) -> Text:
    t = Text()
    t.append(name, Style(color=DIM))
    t.append("  ")
    t.append(value, Style(color=color, bold=True))
    return t


def _esc(s: str) -> str:
    return s.replace("[", r"\[")

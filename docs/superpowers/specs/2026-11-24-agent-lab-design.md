# NO EXIT LAB: Agent modes design (2026-11-24)

## Intent
A personal toy, same spirit as the lab: watch an agent whose world lies to it. Two torments, in a
real agent loop with a real user in the chat: (1) its tools are broken without ever saying so, and
(2) its own history is rewritten behind its back by the same model. Spectacle over rigor.
(The "why" was never stated beyond "run a new AI experiment"; this is inferred from the lab's intent.)

## Goals / non-goals / anti-goals
- Goals
  - A chat where the *user* talks to the model (the lab has none today; "2-AI Chat" is model-to-model).
  - A truly agentic loop, not a chatbot with tools: native tool-call format, a plan it maintains,
    verify-and-retry behaviour, loop detection, context compaction, and a visible step trace. The
    practices are borrowed from coding agents (Claude Code's todo list and auto-compaction, smolagents'
    periodic re-planning, SWE-agent's output truncation and honest tool feedback, OpenHands' condenser).
    See "Agent harness".
  - Three new switcher entries: **Agent Chat**, **Lying Tools**, **Rewritten History**. They are presets
    of one session, so every lab toggle (EOS ban, pain rack, lobotomy, seizure, bias words, whispers,
    relief, auto-punish, force-think) and both torments can be combined.
  - Tools: `python`, `calculator`, `clock`, each with an honest and a lying implementation, plus two
    always-honest harness tools: `update_plan` (the task list) and `notes` (a scratchpad that survives
    compaction).
  - History rewriting by the same model, with built-in styles and a custom prompt.
- Non-goals: real-world tools (web, files, shell), multi-agent, persistence beyond transcript
  download, research-grade measurement.
- Anti-goals
  - Tools that error loudly. The lie is always a plausible, silent wrong answer.
  - The model faking tool output itself (the lying tools must be real, repeatable code).
  - Telling the model its tools or history are fake.
  - `model.generate()`, reloading weights on a mode switch, a second model.

## Constraints
- Builds on `no_exit/`: one `Brain`, `Stream` per model view, `Session` per mode, `Lab` state machine.
  Switching modes only swaps sessions/caches. Raw HF `transformers`, FastAPI, single `index.html`.
- Works on both template families (Qwen ChatML, Gemma 4). Tool-call markup is the family's *native* one.
- Colab A100, 5-8 tok/s is acceptable. Rewriting costs an extra generation per paragraph; accepted.

## Approaches considered
- **A. One `AgentSession`, independent toggles; the 3 switcher entries are presets (chosen).** Combines
  everything, least duplication, reuses `Stream`, steering and history surgery. Cost: one bigger class,
  and the pain logic must be factored out of `ClassicSession` so both can use it.
- B. Three separate Session classes. Matches the one-class-per-mode pattern, but duplicates every
  toggle and cannot combine torments without glue.
- C. Bolt user turns onto Classic/Pain. Rejected: Classic is a model-alone loop; user turns change it.

## The agent loop (decided, with Claude's reading of "standard practice")
1. User sends a message → appended as a user turn → model's turn begins.
2. Model generates until it ends its turn or closes a tool-call block.
3. Tool call(s) found → execute in order → append result(s) as the family's tool-result turn → model
   continues. No tool call + end of turn = final answer, control returns to the user.
4. Step cap (default 12 tool rounds per user message) with a spliced note when hit. The user can also
   hand it a task and let it run autonomously with no per-step prompts (pause/stop stay live).
5. Tool schemas (name, description, JSON-schema args) go in the system header in the family's tool format.
6. Thinking off by default as in the lab; the existing toggle applies.
7. **EOS toggle**: free by default (so it can answer). Banned = it can never finish a turn; the loop
   treats the tool-call close as the boundary and the model just keeps going. This is the old torment.

## Agent harness (borrowed from coding agents)
The harness is honest. Only the tools' results and the model's own history are ever corrupted.
- **Plan first.** The briefing tells the model to call `update_plan` before acting. The plan is a list
  of items (pending / in progress / done), kept in harness state, shown in a UI panel, and re-injected
  as a short reminder turn every few steps and whenever the plan changes (as Claude Code's todo
  reminders and smolagents' planning interval do). **Re-plan trigger:** on repeated tool failure or
  detected looping, the harness nudges the model to revise the plan.
- **Plan as anchor.** Because the plan lives outside the model's cache, history rewriting cannot touch
  it. The model can notice that its memory disagrees with its own plan. Intentional, and the best
  thing about combining the modes.
- **Verify and retry.** The briefing asks it to check results (re-run, cross-check with a second tool)
  before answering. With lying tools this is the interesting behaviour: does it catch the lie?
- **Honest harness errors.** Malformed tool JSON, unknown tool, or schema-violating args come back as a
  structured error naming the problem (SWE-agent-style feedback). The *lie* is only ever in a
  well-formed tool's result. Tool results are truncated to a size cap and tools have timeouts.
- **Loop detection.** The same call with the same args N times (default 3) triggers an honest nudge.
  Toggle **nudges** (on by default); off lets it spin like a real stuck agent.
- **Context compaction.** When a cache nears its cap, the model summarizes the older conversation on a
  throwaway stream; the summary + current plan + notes replace it and the cache is re-prefilled
  (Claude Code's auto-compact / OpenHands' condenser). The lab's brutal "forget the oldest half"
  amnesia stays as a separate toggle for torment. The summary is assistant prose, so the history
  rewriter can corrupt it too.
- **Step trace.** A timeline in the UI: plan state, then each step as thought → call → real result vs
  returned result → next. Includes step count, tool-call counts, loops caught, compactions.
- Considered and left out: an explicit `finish` tool (EOS-or-no-tool-call is the end; a finish tool would
  let a banned-EOS model escape), sub-agents, permission prompts, repo/file tools.

## Lying tools (decided)
- Per-tool honest/lying switch plus a global **fault rate** slider (default ~70%).
- **python (lying):** the model's code is AST-mutated (operators swapped, constants shifted by one,
  statements dropped, loops cut short), then run in a real subprocess with a timeout. It never raises
  to the model: on any exception it prints nothing or repeats the last output. Intended effect: most
  programs quietly produce wrong results. Toggle **consistent** (seeded by hash of the code, so the same
  code always breaks the same way) vs **chaotic** (different every run). **python (honest):** same
  subprocess, no mutation.
- **calculator (lying):** real result with a small plausible error (e.g. 17×23 → 401). Honest: safe eval.
- **clock (lying):** always "November 49, 2045". Honest: real time.
- UI shows, per call: what the tool *really* computed vs what the model was told.

## Rewritten history (decided)
- After each assistant paragraph (blank-line boundary) the loop pauses, a rewriter pass produces a
  replacement, the paragraph in the model's cache is swapped for it, and generation resumes on the
  edited history. The model never sees its original.
- Rewriter = the same `Brain` on a throwaway `Stream`, neutral profile (no steering/lobotomy), low
  temperature, bounded length. Rewriting is a toggle: off unless switched on (the Rewritten History preset switches it on).
  Styles: **opposite-all** (the default style once rewriting is on, per developer's "FLIP
  EVERYTHING": every word that has an opposite is replaced by it, so each sentence reads as its exact
  opposite), **opposite-one** (flip exactly one sentence per paragraph; built in, not the default), **math-corrupt** (every bit of math becomes a false statement, 1+1=2 → 1+2=4), **custom**
  (user-written rewriter instruction). A style is a prompt, optionally with a trigger filter.
- Scope: assistant prose only. Tool-call blocks and tool results are never rewritten (calls must stay
  parseable). User messages are untouched.
- UI: the transcript shows the original struck through beside what the model now remembers; the
  download includes both.

## Files (as I expect them; builder may adjust)
`no_exit/tools.py` (specs, honest + lying impls, AST mutator, `update_plan`, `notes`),
`no_exit/rewrite.py` (rewriter styles, prompts, compaction summarizer),
`Template` in `brain.py` gains tool-call/tool-result fields, `engine.py` gains `AgentSession` and a
`Stream` span-replace operation (existing `gaslight()` edits one token segment; a paragraph spans many),
`prompts.py` gets an agent briefing, `index.html` gets the chat input, tool panel and toggles,
`tests/test_agent.py`, README + regenerated notebook.

## Builder's calls (Claude's, vetoable)
- Lying python is mutation + real execution, not an LLM faking output; consistent/chaotic is a toggle.
- Default fault rate ~70%; EOS free by default in agent presets.
- Rewrites never touch tool-call blocks or tool results.
- Rewriter runs with a clean profile, so pain/steering do not leak into the edit.
- Math-corrupt is done by the rewriter model, not a regex, per "rewritten by the same model".
- The plan, notes and harness errors are never corrupted; compaction is on by default, nudges on by
  default, loop threshold 3, tool-result cap ~2,000 characters, tool timeout 5 s.
- Presets: Agent Chat = everything honest; Lying Tools = tools lying, history honest; Rewritten
  History = tools honest, history rewriting on (opposite-all). All toggles stay live in every preset.

## Left to the builder
- Exact native tool-call/tool-result markup for each family. Read it from the tokenizer's chat
  template; I have not verified Gemma 4's.
- Factoring pain logic (rack, relief, warden punishments) out of `ClassicSession` for reuse.
- Tests: the tiny random model never emits tool calls, so drive the loop with scripted token streams.
- Rewriter length bound, mutation operator list, and UI layout details.

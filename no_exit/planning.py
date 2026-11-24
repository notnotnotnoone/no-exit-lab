"""The agent's task list and scratchpad. Both live in harness state, outside the model's KV cache,
so nothing that rewrites the model's history can touch them."""

from __future__ import annotations

from dataclasses import dataclass

STATUSES = ("pending", "in_progress", "done")
_MARK = {"pending": " ", "in_progress": "~", "done": "x"}


@dataclass
class PlanItem:
    text: str
    status: str


class PlanState:
    def __init__(self) -> None:
        self.items: list[PlanItem] = []
        self._changed = False

    def update(self, raw_items: object) -> str | None:
        """Replace the whole plan. Returns an error message (and changes nothing) if the input is invalid."""
        if not isinstance(raw_items, list):
            return "items must be a list of {text, status} objects"
        parsed: list[PlanItem] = []
        for k, it in enumerate(raw_items, 1):
            if not isinstance(it, dict) or not isinstance(it.get("text"), str) or not it["text"].strip():
                return f"item {k} needs a non-empty string 'text'"
            status = it.get("status", "pending")
            if status not in STATUSES:
                return f"item {k} has status {status!r}; use one of {', '.join(STATUSES)}"
            parsed.append(PlanItem(it["text"], status))
        if sum(p.status == "in_progress" for p in parsed) > 1:
            return "at most one item may be in_progress at a time"
        self.items = parsed
        self._changed = True
        return None

    def to_dicts(self) -> list[dict]:
        return [{"text": i.text, "status": i.status} for i in self.items]

    def render(self) -> str:
        return "\n".join(f"{n}. [{_MARK[i.status]}] {i.text}" for n, i in enumerate(self.items, 1))

    def take_changed(self) -> bool:
        """True once after each successful update."""
        changed, self._changed = self._changed, False
        return changed


class Notes:
    def __init__(self) -> None:
        self.text = ""

    def write(self, s: str, append: bool = False) -> None:
        self.text = f"{self.text}\n{s}" if append and self.text else s

    def read(self) -> str:
        return self.text


def plan_reminder(plan: PlanState) -> str:
    if not plan.items:
        return "[no plan yet: call update_plan to write one before you act]"
    return f"[plan reminder]\n{plan.render()}"

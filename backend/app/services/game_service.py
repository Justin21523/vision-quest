from typing import Any, Dict


class GameService:
    def __init__(self) -> None:
        self.game_state = self._initial_state()

    def _initial_state(self) -> Dict[str, Any]:
        return {
            "scene": "archive_gate",
            "inventory": ["old_map"],
            "stats": {"health": 100, "mana": 50},
            "history": [],
            "branch_id": "archive-gate-root",
            "choices": ["scan glyph", "query memory", "save branch"],
        }

    async def start_new_game(self, scenario: str, persona: Dict[str, Any]) -> Dict[str, Any]:
        self.game_state = {
            **self._initial_state(),
            "scenario": scenario,
            "persona": persona,
        }
        narrative = (
            f"{persona.get('name', 'Alex')} arrives at the luminous archive gate. "
            "A map, a sealed terminal, and a memory shard are visible. Choose whether to scan, enter, or ask the agent for context."
        )
        self.game_state["history"].append({"role": "ai", "text": narrative})
        return {
            "narrative": narrative,
            "game_state": self.game_state,
            "choices": self.game_state["choices"],
            "demo_mode": True,
        }

    async def take_action(self, action: str) -> Dict[str, Any]:
        text = action.strip() or "wait"
        delta = {"target": "player", "action": "update", "value": {}}
        lower = text.lower()
        if "scan" in lower or "掃描" in text:
            narrative = "The VQA layer scans the gate and reveals a cyan access glyph. The RAG memory links it to an old safety protocol."
            delta["value"] = {"mana": max(0, self.game_state["stats"]["mana"] - 5)}
            self.game_state["branch_id"] = "glyph-scan-review"
            self.game_state["choices"] = ["retrieve safety protocol", "inspect terminal", "open history DAG"]
        elif "enter" in lower or "進入" in text:
            narrative = "You enter the archive. The agent opens a safe route and records a new branch in the journey DAG."
            self.game_state["inventory"].append("access_glyph")
            self.game_state["branch_id"] = "archive-entry-route"
            self.game_state["choices"] = ["save state", "ask curator agent", "continue route"]
        else:
            narrative = "The orchestrator evaluates your action and suggests scanning the environment before committing to a branch."
            self.game_state["choices"] = ["scan environment", "query knowledge", "wait"]

        self._apply_delta(delta)
        self.game_state["history"].append({"role": "player", "text": text})
        self.game_state["history"].append({"role": "ai", "text": narrative})
        return {
            "narrative": narrative,
            "game_state": self.game_state,
            "delta": delta,
            "choices": self.game_state["choices"],
            "branch_id": self.game_state["branch_id"],
        }

    def _apply_delta(self, delta: Dict[str, Any]) -> None:
        if delta.get("target") != "player":
            return
        for key, value in (delta.get("value") or {}).items():
            if key in self.game_state["stats"]:
                self.game_state["stats"][key] = value

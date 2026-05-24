"""
cost_tracker.py
Lightweight token budget enforcer for multi-agent pipelines.
Tracks cumulative token usage per run and raises BudgetExceededError
if the configured maximum is exceeded.

Usage:
    from utils.cost_tracker import TokenBudget

    budget = TokenBudget(max_tokens=50000, model="gpt-4o")
    budget.add(prompt_tokens=1200, completion_tokens=400)
    budget.check()   # raises BudgetExceededError if over limit
    print(budget.summary())
"""

import tiktoken
from dataclasses import dataclass, field
from typing import Optional

# Cost per 1M tokens (USD) — update as pricing changes
MODEL_COSTS = {
    "gpt-4o":          {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":     {"input": 0.15,  "output": 0.60},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307":    {"input": 0.25, "output": 1.25},
}

WARN_THRESHOLD = 0.80   # warn at 80% of budget


class BudgetExceededError(Exception):
    """Raised when token usage exceeds the configured maximum."""
    pass


@dataclass
class TokenBudget:
    max_tokens: int = 50_000
    model: str = "gpt-4o"
    prompt_tokens: int = field(default=0, init=False)
    completion_tokens: int = field(default=0, init=False)
    _warned: bool = field(default=False, init=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        costs = MODEL_COSTS.get(self.model, {"input": 5.0, "output": 15.0})
        return (
            self.prompt_tokens     / 1_000_000 * costs["input"] +
            self.completion_tokens / 1_000_000 * costs["output"]
        )

    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Record token usage from one LLM call."""
        self.prompt_tokens     += prompt_tokens
        self.completion_tokens += completion_tokens

        usage_pct = self.total_tokens / self.max_tokens
        if usage_pct >= WARN_THRESHOLD and not self._warned:
            self._warned = True
            print(f"[TokenBudget] WARNING: {usage_pct:.0%} of budget used "
                  f"({self.total_tokens:,}/{self.max_tokens:,} tokens). "
                  f"Est. cost: ${self.estimated_cost_usd:.4f}")

    def check(self):
        """Raise BudgetExceededError if over limit."""
        if self.total_tokens > self.max_tokens:
            raise BudgetExceededError(
                f"Token budget exceeded: {self.total_tokens:,} > {self.max_tokens:,} tokens. "
                f"Estimated cost: ${self.estimated_cost_usd:.4f}. "
                f"Increase max_tokens or reduce agent complexity."
            )

    def summary(self) -> str:
        return (
            f"Token usage: {self.total_tokens:,}/{self.max_tokens:,} "
            f"({self.total_tokens/self.max_tokens:.0%}) | "
            f"Prompt: {self.prompt_tokens:,} | "
            f"Completion: {self.completion_tokens:,} | "
            f"Est. cost: ${self.estimated_cost_usd:.4f}"
        )


def track_tokens(budget: TokenBudget):
    """
    Decorator that wraps an OpenAI completion call and automatically
    records token usage to the provided budget.

    Usage:
        @track_tokens(budget)
        def call_llm(messages):
            return client.chat.completions.create(model="gpt-4o", messages=messages)
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            response = fn(*args, **kwargs)
            if hasattr(response, "usage") and response.usage:
                budget.add(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
                budget.check()
            return response
        return wrapper
    return decorator

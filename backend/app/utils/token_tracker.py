import logging
from typing import Dict
from ..core.config import settings

logger = logging.getLogger(__name__)

class CostLimitExceededError(Exception):
    """Raised when the LLM usage exceeds the configured limits."""
    pass

class TokenTracker:
    """
    Centralized Token Tracker to monitor LLM usage and calculate USD cost across the session.
    """

    # Estimated Pricing per 1 Million tokens (USD)
    # Reference: https://ai.google.dev/pricing  |  https://openai.com/pricing
    PRICING = {
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-preview": {"input": 0.075, "output": 0.30},
        "gemini-3-flash": {"input": 0.10, "output": 0.40},
        "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
        "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
        "gemini-1.5-pro-preview": {"input": 3.50, "output": 10.50},
        "gemini-3-pro": {"input": 5.00, "output": 15.00},
        "gemini-3-pro-preview": {"input": 5.00, "output": 15.00},
        # OpenAI models
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-5.4": {"input": 2.00, "output": 8.00},
        "gpt-5.4-mini": {"input": 0.15, "output": 0.60},
        "gpt-5.4-pro": {"input": 10.00, "output": 30.00},
        "o3": {"input": 10.00, "output": 40.00},
        "o3-mini": {"input": 1.10, "output": 4.40},
        "o4-mini": {"input": 1.10, "output": 4.40},
    }

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the tracker counters for a new session or document run."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0

    def add_usage(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Add tokens from a single LLM API call and calculate incremental cost."""
        self.call_count += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        # Calculate cost
        pricing = self._get_pricing(model)
        incremental_cost = ((prompt_tokens / 1_000_000) * pricing["input"]) + \
                           ((completion_tokens / 1_000_000) * pricing["output"])
        
        self.total_cost_usd += incremental_cost

        # Log incremental usage at debug level to avoid spam, but useful for profiling
        logger.debug(f"LLM Call [{model}]: {prompt_tokens} prompt | {completion_tokens} completion | Cost: ${incremental_cost:.5f}")

        self._check_limits()

    def _get_pricing(self, model: str) -> Dict[str, float]:
        """Resolve pricing for the given model, defaulting to flash pricing if unknown."""
        model_lower = model.lower()
        for key, price in self.PRICING.items():
            if key in model_lower:
                return price
        # Default to standard gemini-1.5-flash pricing if unknown model string
        return {"input": 0.075, "output": 0.30}

    def _check_limits(self):
        """Enforce strict token and cost limits across the session."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        # Check Token Limit
        max_tokens = getattr(settings, "MAX_SESSION_TOKENS", 0)
        if max_tokens > 0 and total_tokens > max_tokens:
            logger.error(f"Token limit exceeded! Used {total_tokens} tokens (Limit: {max_tokens})")
            raise CostLimitExceededError(f"Session Token Limit Exceeded: {total_tokens}/{max_tokens} tokens used.")

        # Check Cost Limit
        max_cost = getattr(settings, "MAX_SESSION_COST_USD", 0.0)
        if max_cost > 0.0 and self.total_cost_usd > max_cost:
            logger.error(f"Cost limit exceeded! Used ${self.total_cost_usd:.4f} USD (Limit: ${max_cost:.4f} USD)")
            raise CostLimitExceededError(f"Session Cost Limit Exceeded: ${self.total_cost_usd:.4f}/${max_cost:.4f} USD used.")

    def get_summary(self) -> str:
        """Return a formatted string summarizing the session's LLM consumption."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        return (f"💰 LLM Session Cost: ${self.total_cost_usd:.5f} USD "
                f"({self.call_count} calls | {total_tokens:,} total tokens | "
                f"Prompt: {self.total_prompt_tokens:,} | Completion: {self.total_completion_tokens:,})")

# Global singleton
token_tracker = TokenTracker()

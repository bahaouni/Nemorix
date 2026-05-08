"""Predictive prefetcher — anticipates which agents will wake up next."""

from __future__ import annotations
from typing import List
from nemorix.core.agent import AgentMemoryObject


class PredictivePrefetcher:
    """Simple prediction based on historical activation frequency."""

    def __init__(self, prefetch_threshold_s: float = 60.0):
        self.prefetch_threshold_s = prefetch_threshold_s

    def predict_next_activation(
        self, agent: AgentMemoryObject, current_time: float
    ) -> float:
        """Estimate seconds until next activation based on activation_probability."""
        if agent.activation_probability <= 0:
            return float("inf")
        expected_interval = 1.0 / agent.activation_probability  # in time steps
        return expected_interval * 60.0  # convert to seconds (60s per step)

    def get_prefetch_candidates(
        self, agents: List[AgentMemoryObject], current_time: float
    ) -> List[str]:
        """Return agent IDs that should be prefetched to a warmer tier."""
        candidates = []
        for agent in agents:
            if agent.state in ("sleeping", "suspended"):
                predicted = self.predict_next_activation(agent, current_time)
                if predicted < self.prefetch_threshold_s:
                    candidates.append(agent.agent_id)
        return candidates

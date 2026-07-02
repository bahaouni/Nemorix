"""Workload generator — creates agents with realistic KV-cache sizes and activation patterns."""

from __future__ import annotations
import random
import uuid
from typing import List
from nemorix.core.kv_block import KVBlock
from nemorix.core.agent import AgentMemoryObject


class ModelConfig:
    """LLM model configuration for KV-cache size calculation."""

    def __init__(
        self,
        name: str = "Llama-3-70B",
        num_layers: int = 80,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        dtype_bytes: int = 2,  # FP16
    ):
        self.name = name
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype_bytes = dtype_bytes

    def kv_bytes_per_token_per_layer(self) -> int:
        """K + V per token per layer."""
        return 2 * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def total_kv_bytes(self, num_tokens: int) -> int:
        return num_tokens * self.kv_bytes_per_token_per_layer() * self.num_layers

    def kv_bytes_per_layer(self, num_tokens: int) -> int:
        return num_tokens * self.kv_bytes_per_token_per_layer()


LLAMA_70B = ModelConfig("Llama-3-70B", 80, 8, 128, 2)
LLAMA_8B = ModelConfig("Llama-3-8B", 32, 8, 128, 2)


class WorkloadGenerator:
    def __init__(
        self,
        model: ModelConfig | None = None,
        seed: int = 42,
    ):
        self.model = model or LLAMA_70B
        self.rng = random.Random(seed)

    def create_agent(
        self,
        context_tokens: int = 65536,
        priority: int | None = None,
        activation_prob: float | None = None,
    ) -> AgentMemoryObject:
        agent_id = uuid.UUID(int=self.rng.getrandbits(128)).hex[:12]
        if priority is None:
            priority = self.rng.randint(1, 9)
        if activation_prob is None:
            activation_prob = self.rng.uniform(0.02, 0.25)

        blocks: List[KVBlock] = []
        for layer in range(self.model.num_layers):
            size = self.model.kv_bytes_per_layer(context_tokens)
            attn_score = self.rng.uniform(0.1, 0.9)
            # System prompt layers (first 5) and final layers have higher attention
            if layer < 5 or layer > self.model.num_layers - 5:
                attn_score = self.rng.uniform(0.6, 1.0)
            blocks.append(
                KVBlock(
                    block_id=f"{agent_id}_L{layer:03d}",
                    agent_id=agent_id,
                    layer_idx=layer,
                    num_tokens=context_tokens,
                    size_bytes=size,
                    dtype="fp16",
                    importance_score=attn_score,
                    tier="ssd",  # start cold
                )
            )

        return AgentMemoryObject(
            agent_id=agent_id,
            blocks=blocks,
            state="suspended",
            priority=priority,
            total_context_tokens=context_tokens,
            activation_probability=activation_prob,
        )

    def create_workload(
        self,
        num_agents: int = 50,
        context_tokens: int = 65536,
    ) -> List[AgentMemoryObject]:
        agents = []
        for _ in range(num_agents):
            tokens = self.rng.randint(
                context_tokens // 2, context_tokens
            )
            agents.append(self.create_agent(context_tokens=tokens))
        return agents

    def generate_activations(
        self, agents: List[AgentMemoryObject], time_step: int
    ) -> List[str]:
        """Return list of agent_ids that should be activated this step."""
        activated = []
        for agent in agents:
            if self.rng.random() < agent.activation_probability:
                activated.append(agent.agent_id)
        return activated

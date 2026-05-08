"""FastAPI server for Nemorix agent management (requires: pip install fastapi uvicorn)."""

from __future__ import annotations
import time

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install FastAPI: pip install fastapi uvicorn pydantic")

from nemorix.core.tier_manager import MemoryTierManager
from nemorix.core.scheduler import AgentScheduler
from nemorix.policies.semantic import SemanticEvictionPolicy
from nemorix.simulation.workload import WorkloadGenerator

app = FastAPI(title="Nemorix", description="Virtual Memory for AI Agents")

# Global state
policy = SemanticEvictionPolicy()
tier_mgr = MemoryTierManager()
scheduler = AgentScheduler(tier_mgr, policy)
workload_gen = WorkloadGenerator()


class CreateAgentBody(BaseModel):
    context_tokens: int = 65536
    priority: int = 5


@app.post("/agents")
def create_agent(body: CreateAgentBody):
    agent = workload_gen.create_agent(body.context_tokens, body.priority)
    policy.set_agent_priority(agent.agent_id, agent.priority)
    scheduler.register_agent(agent)
    for block in agent.blocks:
        tier_mgr.get_tier("ssd").allocate(block.block_id, block.size_bytes)
    return {"agent_id": agent.agent_id, "size_mb": round(agent.total_size_mb, 2)}


@app.post("/agents/{agent_id}/resume")
def resume_agent(agent_id: str):
    if agent_id not in scheduler.agents:
        raise HTTPException(404, "Agent not found")
    latency = scheduler.activate_agent(agent_id, time.time())
    agent = scheduler.agents[agent_id]
    return {
        "agent_id": agent_id,
        "state": agent.state,
        "resume_latency_ms": round(latency, 2),
    }


@app.post("/agents/{agent_id}/pause")
def pause_agent(agent_id: str, tier: str = "cxl"):
    if agent_id not in scheduler.agents:
        raise HTTPException(404, "Agent not found")
    scheduler.deactivate_agent(agent_id, tier, time.time())
    agent = scheduler.agents[agent_id]
    return {"agent_id": agent_id, "state": agent.state, "tier": tier}


@app.get("/agents")
def list_agents():
    agents = []
    for a in scheduler.agents.values():
        agents.append({
            "agent_id": a.agent_id,
            "state": a.state,
            "tier": a.primary_tier,
            "size_mb": round(a.total_size_mb, 2),
            "resumes": a.resume_count,
        })
    return {"agents": agents, "total": len(agents)}


@app.get("/metrics")
def get_metrics():
    tiers = {}
    for name, t in tier_mgr.tiers.items():
        tiers[name] = {
            "capacity_gb": round(t.capacity_bytes / 1024**3, 1),
            "used_gb": round(t.used_bytes / 1024**3, 1),
            "utilization": round(t.utilization * 100, 1),
        }
    return {
        "tiers": tiers,
        "states": scheduler.get_state_counts(),
        "cost_per_hour": round(tier_mgr.total_cost_per_hour(), 4),
    }

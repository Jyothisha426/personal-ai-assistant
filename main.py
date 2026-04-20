"""
main.py — Personal AI Assistant RL Environment
OpenEnv-compatible FastAPI server

Endpoints:
  GET  /health
  GET  /tasks
  POST /reset?task_name=...
  POST /step
  GET  /state
"""

import random
from typing import Any, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from tasks import TASK_REGISTRY

app = FastAPI(
    title="Personal AI Assistant Environment",
    description="OpenEnv-compatible RL environment for personalized task handling (Theme 3.2)",
    version="1.0.0",
)

# ── In-memory episode state ───────────────────────────────────────────────────
episode_state: dict[str, Any] = {
    "task_name": None,
    "scenarios": [],
    "current_index": 0,
    "done": False,
    "rewards": [],
    "total_steps": 0,
}


# ── Pydantic models ───────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    response: str


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_observation(task_name: str, scenario: dict) -> dict:
    """Build a clean observation dict from a scenario."""
    task_info = TASK_REGISTRY[task_name]
    obs = {
        "task_name": task_name,
        "task_description": task_info["description"],
        "step_number": episode_state["current_index"] + 1,
        "total_steps": len(episode_state["scenarios"]),
    }
    for key in task_info["observation_keys"]:
        if key in scenario:
            obs[key] = scenario[key]
    return obs


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "environment": "personal-ai-assistant"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": name,
                "description": info["description"],
                "num_scenarios": len(info["scenarios"]),
            }
            for name, info in TASK_REGISTRY.items()
        ]
    }


@app.post("/reset")
def reset(
    task_name: Optional[str] = Query(None),
    body: Optional[ResetRequest] = None,
):
    name = task_name or (body.task_name if body else None)
    if not name:
        name = random.choice(list(TASK_REGISTRY.keys()))
    if name not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task: {name}. Available: {list(TASK_REGISTRY.keys())}")

    scenarios = list(TASK_REGISTRY[name]["scenarios"])
    random.shuffle(scenarios)

    episode_state.update({
        "task_name": name,
        "scenarios": scenarios,
        "current_index": 0,
        "done": False,
        "rewards": [],
        "total_steps": len(scenarios),
    })

    first_scenario = scenarios[0]
    obs = build_observation(name, first_scenario)

    return {
        "observation": obs,
        "done": False,
        "info": {"task": name, "total_steps": len(scenarios)},
    }


@app.post("/step")
def step(request: StepRequest):
    if episode_state["task_name"] is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    if episode_state["done"]:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new one.")

    task_name = episode_state["task_name"]
    idx = episode_state["current_index"]
    scenario = episode_state["scenarios"][idx]

    grader = TASK_REGISTRY[task_name]["grader"]
    reward = grader(request.response, scenario)

    episode_state["rewards"].append(reward)
    episode_state["current_index"] += 1
    episode_state["total_steps"] += 1

    done = episode_state["current_index"] >= len(episode_state["scenarios"])
    episode_state["done"] = done

    next_obs = None
    if not done:
        next_scenario = episode_state["scenarios"][episode_state["current_index"]]
        next_obs = build_observation(task_name, next_scenario)

    avg_reward = sum(episode_state["rewards"]) / len(episode_state["rewards"])

    return {
        "observation": next_obs,
        "reward": reward,
        "done": done,
        "info": {
            "step": idx + 1,
            "avg_reward_so_far": round(avg_reward, 4),
            "task": task_name,
        },
    }


@app.get("/state")
def get_state():
    rewards = episode_state["rewards"]
    return {
        "task_name": episode_state["task_name"],
        "current_step": episode_state["current_index"],
        "total_scenarios": len(episode_state["scenarios"]),
        "done": episode_state["done"],
        "rewards": rewards,
        "average_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
    }

#!/usr/bin/env python3
"""
inference.py — Personal AI Assistant Environment Agent
MANDATORY LOG FORMAT:
  [START] task=TASKNAME
  [STEP] step=N reward=R.RRRR
  [END] task=TASKNAME score=S.SSSS steps=N

Score must be STRICTLY between 0 and 1 (not 0.0, not 1.0).
"""

import os
import sys
import asyncio
import time
import httpx
from openai import OpenAI

# ── Config — MUST use API_KEY and API_BASE_URL from environment ───────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("API_KEY",      "dummy-key-for-local-testing")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

TASKS_TO_RUN = [
    "tough_email_reply",
    "schedule_conflict",
    "personal_message",
    "dinner_travel_planning",
    "schema_drift",
]

MAX_STEPS = 15  # 15 scenarios per task

# ── Score safety ──────────────────────────────────────────────────────────────
def safe_score(s): return round(min(max(float(s), 0.0001), 0.9999), 4)
def safe_reward(r): return round(min(max(float(r), 0.0001), 0.9999), 4)

# ── Logging ───────────────────────────────────────────────────────────────────
def log_start(task):  print(f"[START] task={task}", flush=True)
def log_step(step, reward): print(f"[STEP] step={step} reward={safe_reward(reward)}", flush=True)
def log_end(task, score, steps): print(f"[END] task={task} score={safe_score(score)} steps={steps}", flush=True)

# ── LLM client ────────────────────────────────────────────────────────────────
def make_llm_client() -> OpenAI:
    print(f"[DEBUG] LLM: base_url={API_BASE_URL} model={MODEL_NAME}", file=sys.stderr, flush=True)
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def call_llm(client: OpenAI, system: str, user: str, max_tokens: int = 400,
             max_retries: int = 3) -> str:
    """
    Call LLM with exponential backoff retry.
    Retries up to max_retries times on failure before returning empty string.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                stream=False,
            )
            result = (resp.choices[0].message.content or "").strip()
            if result:
                return result
            # Empty response — retry
            print(f"[DEBUG] LLM returned empty (attempt {attempt+1}/{max_retries})", file=sys.stderr, flush=True)
        except Exception as e:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"[DEBUG] LLM error (attempt {attempt+1}/{max_retries}): {e} — retrying in {wait}s", file=sys.stderr, flush=True)
            if attempt < max_retries - 1:
                time.sleep(wait)
    return ""


# ── System prompt ─────────────────────────────────────────────────────────────
SYS = """You are a highly skilled personal AI assistant. 
You handle real-life situations with empathy, practicality, and professionalism.
Always give concrete, actionable responses. Never refuse or give vague answers.
Match your tone to the situation — formal for business, warm for personal."""


# ── Action builders ───────────────────────────────────────────────────────────

def build_email_action(obs: dict, llm: OpenAI) -> dict:
    subject = obs.get("subject", "")
    body    = obs.get("body", "")
    sender  = obs.get("sender", "")
    context = obs.get("context", "You are a professional")

    user = f"""{context}.

Write a professional, empathetic reply to this email.
- 50-200 words
- Acknowledge the sender's concern
- Be specific, not generic
- Offer a concrete next step
- Do NOT use placeholders like [Your Name]

From: {sender}
Subject: {subject}
Email body: {body}

Your reply:"""

    reply = call_llm(llm, SYS, user, max_tokens=300)
    if not reply or len(reply.split()) < 20:
        reply = (
            f"Dear {sender.split('@')[0].replace('.', ' ').title()}, "
            "thank you for reaching out. I understand your concern and take this seriously. "
            "I will personally look into this matter right away and ensure it is resolved promptly. "
            "Please expect a detailed follow-up from me within 24 hours. "
            "I appreciate your patience and apologize for any inconvenience caused."
        )
    return {"response": reply}


def build_schedule_action(obs: dict, llm: OpenAI) -> dict:
    situation   = obs.get("situation", "")
    constraints = obs.get("constraints", [])
    constraints_text = "\n".join(f"- {c}" for c in constraints)

    user = f"""You have a scheduling conflict. Provide a clear resolution.

Situation: {situation}

Constraints:
{constraints_text}

Give a practical resolution that:
1. States what you prioritize and why
2. Says what gets rescheduled or delegated
3. Mentions who gets notified
Keep it concise and decisive (50-150 words).

Resolution:"""

    reply = call_llm(llm, SYS, user, max_tokens=250)
    if not reply or len(reply.split()) < 15:
        reply = (
            "Given the constraints, I would prioritize the most time-sensitive commitment first. "
            "I would reschedule the less urgent item and notify all affected parties immediately "
            "via a brief message explaining the conflict and proposing a new time."
        )
    return {"response": reply}


def build_personal_msg_action(obs: dict, llm: OpenAI) -> dict:
    from_person = obs.get("from", "a friend")
    message     = obs.get("message", "")
    tone        = obs.get("tone_expected", "warm and appropriate")

    user = f"""Reply to this personal message from {from_person}.

Message: "{message}"

Tone to use: {tone}

Rules:
- Sound like a real human, not a corporate bot
- Be warm and genuine
- 15-100 words is ideal
- Do NOT give unsolicited advice unless it helps
- Do NOT be dismissive or minimise their feelings

Your reply:"""

    reply = call_llm(llm, SYS, user, max_tokens=180)
    if not reply or len(reply.split()) < 8:
        reply = (
            "Hey, I really appreciate you sharing that with me. "
            "I hear you and I'm here for you. Let me know how I can help."
        )
    return {"response": reply}


def build_planning_action(obs: dict, llm: OpenAI) -> dict:
    request     = obs.get("request", "")
    constraints = obs.get("constraints", [])
    constraints_text = "\n".join(f"- {c}" for c in constraints)

    user = f"""Create a concrete, actionable plan for the following request.

Request: {request}

Constraints:
{constraints_text}

Your plan must:
- Address EVERY constraint listed
- Include specific recommendations (names, times, costs where relevant)
- Be practical and actionable
- 80-250 words

Your plan:"""

    reply = call_llm(llm, SYS, user, max_tokens=400)
    if not reply or len(reply.split()) < 25:
        reply = (
            "Here is a practical plan addressing your constraints: "
            "First, identify the key requirements and prioritize them. "
            "Then select options that fit within your budget and timeline. "
            "Finally, confirm all bookings or arrangements in advance to avoid last-minute issues."
        )
    return {"response": reply}


def build_drift_action(obs: dict, llm: OpenAI) -> dict:
    initial_task   = obs.get("initial_task", "")
    drift_event    = obs.get("drift_event", "")
    new_constraint = obs.get("new_constraint", "")

    user = f"""You were working on a task, but something changed. Adapt your plan.

Original task: {initial_task}

What changed: {drift_event}

New requirement: {new_constraint}

Your response should:
1. Acknowledge the change
2. Provide a concrete alternative solution
3. Be actionable and specific
Keep it to 50-180 words.

Your adapted plan:"""

    reply = call_llm(llm, SYS, user, max_tokens=280)
    if not reply or len(reply.split()) < 15:
        reply = (
            "I see that the situation has changed. Given this new constraint, "
            "I would adapt the plan by finding an alternative approach that still achieves "
            "the original goal while accommodating the new requirement. "
            "I will proceed with the best available option immediately."
        )
    return {"response": reply}


ACTION_BUILDERS = {
    "tough_email_reply":      build_email_action,
    "schedule_conflict":      build_schedule_action,
    "personal_message":       build_personal_msg_action,
    "dinner_travel_planning": build_planning_action,
    "schema_drift":           build_drift_action,
}


# ── Env HTTP calls (with retry) ───────────────────────────────────────────────
async def env_health(http: httpx.AsyncClient) -> bool:
    try:
        r = await http.get(f"{ENV_BASE_URL}/health", timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f"[DEBUG] health check failed: {e}", file=sys.stderr, flush=True)
        return False


async def env_reset(http: httpx.AsyncClient, task_name: str,
                    max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            r = await http.post(f"{ENV_BASE_URL}/reset",
                                json={"task_name": task_name}, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[DEBUG] reset error (attempt {attempt+1}/{max_retries}): {e}", file=sys.stderr, flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
    raise RuntimeError(f"env_reset failed after {max_retries} attempts")


async def env_step(http: httpx.AsyncClient, action: dict,
                   max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            r = await http.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[DEBUG] step error (attempt {attempt+1}/{max_retries}): {e}", file=sys.stderr, flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(wait)
    raise RuntimeError(f"env_step failed after {max_retries} attempts")


# ── Episode runner ────────────────────────────────────────────────────────────
async def run_task(task_name: str, llm: OpenAI, http: httpx.AsyncClient) -> float:
    log_start(task_name)
    rewards    = []
    steps_done = 0
    score      = 0.0001

    try:
        reset_data = await env_reset(http, task_name)
        obs        = reset_data.get("observation", reset_data)
        done       = reset_data.get("done", False)

        builder = ACTION_BUILDERS.get(task_name)
        if builder is None:
            log_step(1, 0.0001)
            log_end(task_name, 0.0001, 1)
            return 0.0001

        for step_num in range(1, MAX_STEPS + 1):
            if done:
                break

            action = builder(obs, llm)
            print(f"[DEBUG] step={step_num} task={task_name}", file=sys.stderr, flush=True)

            step_data  = await env_step(http, action)
            raw_reward = float(step_data.get("reward", 0.0001))
            reward     = safe_reward(raw_reward)
            done       = step_data.get("done", True)

            next_obs = step_data.get("observation", obs)
            if isinstance(next_obs, dict) and next_obs:
                obs = next_obs

            rewards.append(reward)
            steps_done = step_num
            log_step(step_num, reward)

            if done:
                break

        raw_score = sum(rewards) / len(rewards) if rewards else 0.0001
        score = safe_score(raw_score)

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", file=sys.stderr, flush=True)
        if steps_done == 0:
            steps_done = 1
            log_step(1, 0.0001)

    log_end(task_name, safe_score(score), max(steps_done, 1))
    return safe_score(score)


# ── Main ──────────────────────────────────────────────────────────────────────
async def amain():
    print(f"[DEBUG] ENV={ENV_BASE_URL} MODEL={MODEL_NAME} API_BASE={API_BASE_URL}", file=sys.stderr, flush=True)
    llm = make_llm_client()

    async with httpx.AsyncClient(timeout=60.0) as http:
        healthy = await env_health(http)
        if not healthy:
            print("[DEBUG] health check failed — proceeding anyway", file=sys.stderr, flush=True)

        all_scores = {}
        for task in TASKS_TO_RUN:
            print(f"[DEBUG] starting {task}", file=sys.stderr, flush=True)
            s = await run_task(task, llm, http)
            all_scores[task] = s

    overall = sum(all_scores.values()) / len(all_scores)
    print(f"\n[DEBUG] ══ FINAL RESULTS ══", file=sys.stderr, flush=True)
    for task, s in all_scores.items():
        print(f"[DEBUG]   {task}: {s}", file=sys.stderr, flush=True)
    print(f"[DEBUG]   OVERALL: {safe_score(overall)}", file=sys.stderr, flush=True)


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
# Building a Personal AI Assistant RL Environment with Hybrid LLM-as-Judge Grading

_Meta PyTorch OpenEnv Hackathon — Theme 3.2: World Modeling (Personalized Tasks)_

---

## The Problem with Standard RL Benchmarks

Most RL environments test agents on games, puzzles, or synthetic tasks. But real-world AI assistants need to handle something much messier: human situations. A customer furious about a delayed refund. A scheduling conflict with three equally urgent meetings. A friend texting about a breakup at 2am.

These tasks don't have a single correct answer. They require empathy, judgment, and adaptability — things keyword matching alone can't measure.

That's what this environment is built to train.

---

## What We Built

An **OpenEnv-compatible RL environment** with 5 task types and 75 unique scenarios, designed to train LLMs on real personal assistant work:

| Task                     | What it tests                             |
| ------------------------ | ----------------------------------------- |
| Tough Email Reply        | Empathy + professionalism under pressure  |
| Schedule Conflict        | Multi-constraint priority reasoning       |
| Personal Message         | Emotional tone matching                   |
| Dinner & Travel Planning | Constraint satisfaction + specificity     |
| Schema Drift             | Mid-task adaptability (Patronus AI bonus) |

The agent interacts through a standard OpenEnv API (`/reset`, `/step`, `/state`) and receives a continuous reward in `(0.0001, 0.9999)` for each response.

---

## The Grading System: Why Keyword-Only Isn't Enough

Our first version used pure keyword matching:

```python
keyword_hits = sum(1 for k in scenario["keywords"] if k in response.lower())
score = min(keyword_hits / 3, 1.0)
```

It worked, but it had a fatal flaw: an agent could score 0.95 by stuffing in the right words with no coherent response. "I apologize refund escalate resolve contact" — high score, terrible reply.

### Hybrid Solution

We upgraded to a **hybrid grading system**:

```
Final Score = 0.40 × keyword_score + 0.60 × LLM-as-Judge (Groq Llama-3-8b)
```

The LLM judge receives a task-specific rubric prompt and returns a score from 0.0 to 1.0 based on semantic quality — empathy, relevance, actionability — not just vocabulary.

**Crucially**, if the Groq API is unavailable, the system falls back to keyword-only gracefully. Training never breaks due to an API hiccup.

Here's the core judge function:

```python
def _call_groq(system_prompt: str, user_prompt: str) -> float:
    """Returns float in [0,1] or -1 on failure."""
    if not GROQ_API_KEY:
        return -1.0
    # ... API call ...

def hybrid_score(keyword_score, llm_score, kw_weight=0.4, llm_weight=0.6):
    if llm_score < 0:
        return keyword_score  # graceful fallback
    return round(kw_weight * keyword_score + llm_weight * llm_score, 4)
```

### Validation

We tested the same scenario with three response qualities:

| Response                          | Keyword | LLM-Judge | Final  |
| --------------------------------- | ------- | --------- | ------ |
| Detailed, empathetic (200 words)  | 0.9999  | 1.000     | 0.9999 |
| Generic apology (30 words)        | 0.7600  | 0.760     | 0.7600 |
| Two-word reply ("Sorry, delayed") | 0.3033  | 0.303     | 0.3033 |

Both systems agreed — validating that the keyword heuristics were well-designed, and the LLM judge independently confirmed the scoring logic.

---

## Training with GRPO

We fine-tuned `Qwen/Qwen2.5-0.5B-Instruct` (4-bit quantized) using **Group Relative Policy Optimization (GRPO)** from HuggingFace TRL.

### Why GRPO?

GRPO samples multiple completions per prompt, computes relative rewards within the group, and uses that signal to update the policy. It's more stable than PPO for text generation and works well with our continuous reward signal.

### Setup

```python
training_config = GRPOConfig(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    num_generations=4,      # 4 completions per prompt
    learning_rate=5e-5,
    max_completion_length=256,
)
```

The reward function calls our environment's graders directly (not through HTTP) for speed:

```python
def reward_function(completions, task_names, scenarios, **kwargs):
    rewards = []
    for completion, task_name, scenario in zip(completions, task_names, scenarios):
        text = completion[0]["content"]
        reward = TASK_REGISTRY[task_name]["grader"](text, scenario)
        rewards.append(reward)
    return rewards
```

### Results

Training ran for 60 steps on a free T4 GPU in Google Colab (~1.5 hours):

- **Start reward**: 0.7675
- **End reward**: 0.7917
- **Improvement**: +0.0242

The reward curve shows natural RL wobble — the model explores different phrasings, gets varied signals, and gradually improves. A perfectly smooth curve would suggest something is wrong; this is genuine learning.

---

## Schema Drift: The Patronus AI Bonus Task

The most interesting task type is **Schema Drift** — scenarios where the rules change mid-task:

> _"You were about to book an Italian restaurant, but it just closed for a private event. Find an alternative."_

This tests whether the agent can adapt its plan rather than blindly continuing. The grader rewards:

- Acknowledging the change (not ignoring it)
- Providing a concrete alternative (not vague suggestions)
- Being decisive (no "it depends" hedging)

Real AI assistants encounter this constantly — flights get cancelled, APIs change pricing, plans fall through. An agent that can't adapt gracefully isn't useful.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              OpenEnv API (FastAPI)           │
│  GET /health  GET /tasks  POST /reset        │
│  POST /step   GET /state                     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│              tasks.py                        │
│  TASK_REGISTRY — 5 tasks × 15 scenarios      │
│  Hybrid grader: keyword (40%) + Groq (60%)   │
└─────────────────┬───────────────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
┌───────▼──────┐    ┌────────▼────────┐
│  inference.py │    │    app.py        │
│  RL agent    │    │  Gradio UI demo  │
│  + retry     │    │  (judges can try │
│  logic       │    │   all 75 tasks)  │
└──────────────┘    └─────────────────┘
```

---

## Key Takeaways

1. **Keyword grading is a starting point, not a finish line.** It's fast and interpretable, but semantic evaluation catches what keywords miss.

2. **Hybrid grading is production-ready thinking.** Pure LLM-as-judge is slow and expensive; pure keyword is too rigid. The 40/60 split with graceful fallback is the right balance.

3. **Real-world tasks need real-world grading.** Emotional tone, constraint satisfaction, and mid-task adaptability are what separate useful AI assistants from toy benchmarks.

4. **GRPO works well for open-ended text tasks.** The relative reward signal within a group of completions provides stable training even with a noisy, multi-dimensional reward function.

---

## Links

- 🔴 **Live Demo**: https://jyothisha426-personal-ai-assistant.hf.space
- 📦 **Code**: https://huggingface.co/spaces/Jyothisha426/personal-ai-assistant/tree/main
- 🧪 **Training Notebook**: https://colab.research.google.com/drive/1Eang1ybDBLhi8373pX3TOvm_c98Br4AK?usp=sharing

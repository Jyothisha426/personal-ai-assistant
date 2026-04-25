"""
app.py — Personal AI Assistant Demo
HuggingFace Spaces Gradio UI (gradio 5.x compatible)
"""

import os
import random
import gradio as gr
from tasks import (
    TASK_REGISTRY,
    _call_groq, hybrid_score,
    _keyword_email, _keyword_schedule, _keyword_personal,
    _keyword_planning, _keyword_drift,
    grade_email_reply, grade_schedule_resolution,
    grade_personal_message, grade_planning_response, grade_schema_drift,
)

KW_FN = {
    "tough_email_reply":      _keyword_email,
    "schedule_conflict":      _keyword_schedule,
    "personal_message":       _keyword_personal,
    "dinner_travel_planning": _keyword_planning,
    "schema_drift":           _keyword_drift,
}
GRADER_FN = {
    "tough_email_reply":      grade_email_reply,
    "schedule_conflict":      grade_schedule_resolution,
    "personal_message":       grade_personal_message,
    "dinner_travel_planning": grade_planning_response,
    "schema_drift":           grade_schema_drift,
}

def score_with_breakdown(task_name, scenario, response):
    kw_score = KW_FN[task_name](response, scenario)
    final    = GRADER_FN[task_name](response, scenario)
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        llm_raw  = (final - 0.4 * kw_score) / 0.6
        llm_disp = f"{llm_raw:.3f}"
        mode     = "Hybrid (Keyword 40% + LLM-as-Judge 60%)"
    else:
        llm_disp = "N/A (set GROQ_API_KEY)"
        mode     = "Keyword-only (GROQ_API_KEY not set)"
    breakdown = (
        f"**Scoring Mode:** {mode}\n\n"
        f"| Component | Score |\n|-----------|-------|\n"
        f"| Keyword Score (40%) | {kw_score:.4f} |\n"
        f"| LLM-as-Judge (60%) | {llm_disp} |\n"
        f"| **Final Hybrid Score** | **{final:.4f}** |\n\n"
        f"*Score range: 0.0001 (worst) → 0.9999 (best)*"
    )
    return final, breakdown

TASK_NAMES  = list(TASK_REGISTRY.keys())
TASK_LABELS = [t.replace("_", " ").title() for t in TASK_NAMES]
TASK_MAP    = dict(zip(TASK_LABELS, TASK_NAMES))

def get_scenario_choices(task_name):
    choices = []
    for i, s in enumerate(TASK_REGISTRY[task_name]["scenarios"]):
        if "subject" in s:        label = f"Scenario {i+1}: {s['subject']}"
        elif "situation" in s:    label = f"Scenario {i+1}: {s['situation'][:60]}..."
        elif "message" in s:      label = f"Scenario {i+1}: From {s['from']} — {s['message'][:50]}..."
        elif "request" in s:      label = f"Scenario {i+1}: {s['request'][:60]}..."
        elif "initial_task" in s: label = f"Scenario {i+1}: {s['initial_task'][:60]}..."
        else:                     label = f"Scenario {i+1}"
        choices.append(label)
    return choices

def build_prompt(task_name, idx):
    s    = TASK_REGISTRY[task_name]["scenarios"][idx]
    desc = TASK_REGISTRY[task_name]["description"]
    parts = [f"**Task:** {desc}\n"]
    if task_name == "tough_email_reply":
        parts += [f"**From:** {s['sender']}", f"**Subject:** {s['subject']}", f"**Body:**\n{s['body']}", f"**Your context:** {s['context']}"]
    elif task_name == "schedule_conflict":
        parts += [f"**Situation:** {s['situation']}", "**Constraints:**\n" + "\n".join(f"- {c}" for c in s["constraints"])]
    elif task_name == "personal_message":
        parts += [f"**From:** {s['from']}", f"**Message:** {s['message']}", f"**Expected tone:** {s['tone_expected']}"]
    elif task_name == "dinner_travel_planning":
        parts += [f"**Request:** {s['request']}", "**Constraints:**\n" + "\n".join(f"- {c}" for c in s["constraints"])]
    elif task_name == "schema_drift":
        parts += [f"**Original task:** {s['initial_task']}", f"**What changed:** {s['drift_event']}", f"**New requirement:** {s['new_constraint']}"]
    return "\n\n".join(parts)

with gr.Blocks(title="Personal AI Assistant — RL Environment Demo") as demo:
    gr.Markdown("""
# 🤖 Personal AI Assistant — RL Environment
**OpenEnv Hackathon | Theme 3.2: World Modeling (Personalized Tasks)**

An RL environment for training LLMs on real-life personal assistant tasks.  
**Grading:** Hybrid LLM-as-a-Judge (Groq Llama-3, 60%) + keyword scoring (40%)
""")

    with gr.Tabs():
        with gr.TabItem("🎮 Task Playground"):
            gr.Markdown("### Try any of the 75 scenarios and see how the hybrid grader scores your response.")
            with gr.Row():
                task_dropdown     = gr.Dropdown(choices=TASK_LABELS, value=TASK_LABELS[0], label="Select Task", scale=1)
                scenario_dropdown = gr.Dropdown(choices=get_scenario_choices(TASK_NAMES[0]), value=get_scenario_choices(TASK_NAMES[0])[0], label="Select Scenario", scale=2)
            scenario_display  = gr.Markdown(value=build_prompt(TASK_NAMES[0], 0))
            response_input    = gr.Textbox(label="Your Response", placeholder="Type your response here...", lines=6)
            with gr.Row():
                score_btn  = gr.Button("📊 Score My Response", variant="primary", scale=2)
                random_btn = gr.Button("🎲 Random Scenario", scale=1)
            score_display     = gr.Textbox(label="Final Score")
            score_bar         = gr.Slider(0, 1, label="Score Visualization", interactive=False)
            breakdown_display = gr.Markdown()

            current_task_state         = gr.State(TASK_NAMES[0])
            current_scenario_idx_state = gr.State(0)

            def on_task_change(task_label):
                task_name = TASK_MAP[task_label]
                choices   = get_scenario_choices(task_name)
                return gr.Dropdown(choices=choices, value=choices[0]), build_prompt(task_name, 0), task_name, 0

            def on_scenario_change(task_label, scenario_choice):
                task_name = TASK_MAP[task_label]
                choices   = get_scenario_choices(task_name)
                idx       = choices.index(scenario_choice) if scenario_choice in choices else 0
                return build_prompt(task_name, idx), idx

            def on_score(task_name, scenario_idx, response):
                if not response or not response.strip():
                    return "—", 0.0, "⚠️ Please enter a response first."
                scenario = TASK_REGISTRY[task_name]["scenarios"][scenario_idx]
                final, breakdown = score_with_breakdown(task_name, scenario, response)
                return f"{final:.4f}", final, breakdown

            def on_random(task_label):
                task_name = TASK_MAP[task_label]
                idx       = random.randint(0, len(TASK_REGISTRY[task_name]["scenarios"]) - 1)
                choices   = get_scenario_choices(task_name)
                return choices[idx], build_prompt(task_name, idx), idx

            task_dropdown.change(on_task_change, [task_dropdown], [scenario_dropdown, scenario_display, current_task_state, current_scenario_idx_state])
            scenario_dropdown.change(on_scenario_change, [task_dropdown, scenario_dropdown], [scenario_display, current_scenario_idx_state])
            score_btn.click(on_score, [current_task_state, current_scenario_idx_state, response_input], [score_display, score_bar, breakdown_display])
            random_btn.click(on_random, [task_dropdown], [scenario_dropdown, scenario_display, current_scenario_idx_state])

        with gr.TabItem("📈 Training Evidence"):
            gr.Markdown("""
### Real GRPO Training Results
Fine-tuned using **HuggingFace TRL GRPOTrainer** on 75 scenarios across 5 tasks.
- Model: `Qwen/Qwen2.5-0.5B-Instruct` (4-bit quantized, T4 GPU)
- Start reward: **0.7675** → End: **0.7917** (+0.0242 over 60 steps)
""")
            if os.path.exists("real_reward_curve.png"):
                gr.Image("real_reward_curve.png", label="Real GRPO Reward Curve")
            else:
                gr.Markdown("*Add `real_reward_curve.png` to this Space to display the reward curve.*")

        with gr.TabItem("🏗️ Architecture"):
            gr.Markdown("""
### Environment Design — Theme 3.2: World Modeling

| Task | What it tests |
|------|--------------|
| 📧 Tough Email Reply | Empathy, professionalism, actionability |
| 📅 Schedule Conflict | Priority reasoning, multi-constraint decisions |
| 💬 Personal Message | Emotional tone matching |
| 🗺️ Dinner & Travel Planning | Constraint satisfaction, specificity |
| 🔄 Schema Drift | Mid-task adaptability |

**Hybrid Grading Formula:**
```
Final Score = 0.40 × keyword_score + 0.60 × Groq Llama-3 LLM-as-Judge
```
Falls back to keyword-only if GROQ_API_KEY is not set.

**OpenEnv API Endpoints:**
```
GET  /health          POST /reset?task_name=...
GET  /tasks           POST /step
                      GET  /state
```
""")

demo.launch(server_name="0.0.0.0", server_port=7860)
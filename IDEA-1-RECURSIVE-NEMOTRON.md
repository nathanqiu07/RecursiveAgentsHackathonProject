# Keystone: Recursive Nemotron Physics Self-Improvement

Working name. Idea #1 from `IDEAS.md`, expanded into a build spec. As of Sat Oct 17, 2026, 12:15 PM PDT.

**Deadline assumption:** submissions close at 5:00 PM PDT (00:00 UTC Oct 18), with 1 hour of grace. Confirm with an organizer. Every time below assumes 5:00 PM.

## The pitch

An agent team turns base Nemotron into a better 3D block stacker, with no human labels and nobody at the keyboard. Each round, the agents generate physics tasks at the edge of the model's ability and collect the model's own attempts. A physics simulator grades every attempt. The agents fine-tune on the winners, deploy the new adapter, and check it against a held-out set. They promote it or roll it back, then the new model becomes the worker for the next round. We walk away, and the success curve goes up.

**One-liner for the judges:** "A model that improves itself at intuitive physics, run by a team of agents we didn't touch for two hours."

## Why this fits the theme

| What the event page asks for | Where it lives in Keystone |
|---|---|
| "take a goal and run with it" | The orchestrator gets one goal: raise held-out pass@1 from X% to Y% |
| "make the plan" | The orchestrator picks each next action (curriculum mix, data size, fine-tune settings, promote, rollback, stop) and logs why |
| "spin up sub-agents, delegate work" | The curriculum agent, a pool of 32 to 64 solver workers, the curator, the trainer and the eval runner |
| "communicate with each other" | Typed JSON messages on agent inboxes, all mirrored to `events.jsonl` and shown live |
| "check their own progress" | A fixed held-out set scored after every round, with the scores tracked in the ledger |
| "recover when things break" | Rollback on regression, fine-tune retry, worker respawn, and a restart from saved state after `kill -9` |
| "memory" | The ledger (rounds, adapters, scores, failure types) and the library of verified solutions |
| "long-running inference... after you walk away" | 2:15 to 4:00 PM unattended, with a "human interventions: 0" counter |
| Crusoe Managed Inference + Nemotron | Nemotron on Crusoe for every role, plus Serverless Fine-Tuning and Self-Serve Deployments for the improving model |

## The task: stable stacking

The solver gets a table and a set of rectangular blocks with random sizes and masses. It outputs where to place each block. PyBullet drops them in order and checks whether the structure stands and meets the goal.

Random sizes and masses mean the model can't recite a memorized answer such as the textbook harmonic overhang.

### Goal types and difficulty tiers

| Tier | Blocks | Goal | What makes it hard |
|---|---|---|---|
| T1 | 3 | Tower top at or above a target height | Little: similar masses (ratio up to 2), wide blocks |
| T2 | 5 | Tower top at or above a target height | Mass ratio up to 10, narrow and tall blocks, lower friction |
| T3 | 4 to 6 | Overhang past the table edge at or above a target | The target is 60 to 90% of what a simple greedy heuristic reaches on that scene |
| Unseen | 7 | Height or overhang | Never trained on, used to test generalization |

The curriculum agent controls tier mix, block count and target fraction. It keeps the training mix where the current model succeeds 30 to 70% of the time.

### Solver input (user message)

```json
{
  "goal": {"type": "height", "target_m": 0.18},
  "table": {"friction": 0.5, "edge_x_m": null},
  "blocks": [
    {"id": "A", "size_m": [0.12, 0.05, 0.03], "mass_kg": 0.40},
    {"id": "B", "size_m": [0.04, 0.04, 0.08], "mass_kg": 0.05},
    {"id": "C", "size_m": [0.09, 0.09, 0.02], "mass_kg": 0.30}
  ]
}
```

For overhang goals, the table is a box whose edge sits at `edge_x_m = 0`, and overhang is measured in +x.

### Solver output (assistant message)

```json
{
  "plan": "Heavy wide block at the base, light tall block last so the center of mass stays low.",
  "placements": [
    {"id": "A", "x": 0.0, "y": 0.0, "yaw_deg": 0},
    {"id": "C", "x": 0.0, "y": 0.0, "yaw_deg": 45},
    {"id": "B", "x": 0.01, "y": 0.0, "yaw_deg": 0}
  ]
}
```

- The order of `placements` is the drop order, and choosing it is part of the task.
- z is not an output. The verifier drops each block from 5 mm above the current top.
- `plan` is capped at 40 words, which keeps the training data short.
- Cap `max_tokens` at about 512. If the Nemotron model you use has a reasoning on/off switch, turn it off for solvers (check the model card).

### Verifier

1. Parse the JSON. Invalid JSON, or an id that is missing, duplicated or unknown, counts as a fail with type `parse_error`.
2. For each placement in order:
    1. Spawn the block at (x, y, top + h/2 + 5 mm) with that yaw and zero velocity.
    2. Let it settle for 0.5 s at 240 Hz.
3. After the last block, settle for 3 s.
4. Fail types, used by the curriculum agent:
    - `topple`: a block's z dropped more than 1 cm below where it landed. The first block to move is recorded.
    - `slide`: a block's xy moved more than 2 cm.
    - `fell_off`: a block ended below the table top.
    - `under_target`: the structure stood but missed the goal.
5. Score:
    - Height goals: final top height / target.
    - Overhang goals: max x past the edge / target.
    - Success means no fail type and a score of at least 1.0.
6. Run it in a `multiprocessing` pool of PyBullet DIRECT clients, one per core.

**Speed on this Mac (measured at 12:13 PM):**
- A 5-block episode (0.5 s settle per block, then 3 s) takes about **90 ms on one core**. This Mac has 12 cores.
- One round of 2,400 attempts is therefore about 20 s of verifier time across 12 cores. The simulator is not the bottleneck; LLM calls are.
- `pip install pybullet` built from source here (no prebuilt package for this Python), which took about 3 minutes. Every teammate should start that install now.

### Evaluation protocol

- **Held-out set:** 200 scenes with fixed seeds 10000 to 10199 (70 T1, 70 T2, 60 T3), plus 50 Unseen scenes. They never enter training or the retrieval library.
- **Headline metric:** pass@1 at temperature 0, reported per tier, per round.
- **Generation:** k = 8 samples per scene at temperature 0.8.
- **Honesty column:** the share of each round's training data the model generated itself vs teacher data from Super or Ultra. Judges will ask whether this is self-improvement or distillation, and this column answers it.

## The agent team

```
                         ┌──────────────────────────────┐
     goal ─────────────▶ │ Orchestrator (Nemotron Super)│◀── ledger summary
                         └──┬──────┬──────┬──────┬──────┘
         generate_scenes    │      │      │      │   promote / rollback / stop
              ┌─────────────┘      │      │      └──────────────┐
              ▼                    ▼      ▼                     ▼
      Curriculum agent      Solver pool  Curator          Trainer
      (Super)               (current     (rules + Ultra   (Crusoe fine-tune
      tiers, targets,        model,       repair of near   + deploy wrapper)
      failure focus          32 to 64     misses)               │
              │              workers)       ▲                   ▼
              ▼                    │        │             Eval runner
        scenes.jsonl ──────────────┘        │             (held-out set)
                                   ▼        │                   │
                         PyBullet verifier pool ────────────────┘
                                   │
                                   ▼
             Ledger: events.jsonl · state.json · rounds/ · library/
```

| Agent | Model | Input | Output | Decides |
|---|---|---|---|---|
| Orchestrator | Nemotron 3 Super | Goal plus a ledger summary (last 3 rounds, scores per tier, failure counts, budget) | One action from the action set, with a 1 to 2 sentence rationale | What happens next |
| Curriculum | Nemotron 3 Super | Failure taxonomy of the last round | Generator parameters: tier mix, block counts, mass ranges, target fraction, a "focus" (e.g. heavy-on-light stacks) | Where the frontier is |
| Solver workers | The current champion: base Nano, then the latest adapter | One scene | One placement plan, k times | Nothing; they just attempt |
| Verifier | PyBullet, no LLM | Scene + plan | Pass/fail, score, fail type | Ground truth |
| Curator | Rules + Nemotron 3 Ultra | Graded attempts | Training set: best of k per scene, at most 2 per scene, plus repaired near misses that now pass | What the model learns from |
| Trainer | No LLM (API wrapper) | Dataset id, learning rate, epochs | Adapter id, deployment endpoint, or a failure reason | Retries on failure |
| Eval runner | No LLM | Model id | pass@1 per tier on the held-out set | Nothing; it reports |

### Orchestrator action set

The orchestrator calls these as tools. Hard rules in code override it.

- `generate_scenes(tier_mix, n, focus)`
- `collect(model_id, scene_set, k)`
- `curate(min_score, repair)`
- `finetune(dataset_id, lr, epochs)`
- `evaluate(model_id)`
- `promote(model_id)` or `rollback()`
- `update_library(version)` (in-context path)
- `stop(reason)`

**Hard rules (code, not the LLM):**
- Never promote a model whose held-out pass@1 is more than 2 points below the champion's.
- Retry a fine-tune at most 2 times.
- Stop data generation when the token budget is reached.
- Always evaluate before promoting.

### Messages and memory

- Each agent is an `asyncio` worker with an inbox queue.
- **Message shape:** `{"ts", "from", "to", "type", "payload"}`. Every message is also appended to `ledger/events.jsonl`, which feeds the dashboard's live feed.
- **`state.json`:** written after every orchestrator action. It holds the champion model, round number, pending jobs and dataset ids. `python run.py --resume` continues from it.
- **`rounds/<n>/`:** that round's scenes, attempts, curated set, adapter id, eval results and the orchestrator's rationale.
- **`library/v<n>.jsonl`:** verified solutions for the in-context path.

## The recursion, two ways

### Path A: fine-tuning (the main story)

| Round | Solver | Steps |
|---|---|---|
| 0 | Base Nemotron Nano | Score on the held-out set, then generate and grade 300 scenes × 8 attempts |
| 1 | Base | Curate, fine-tune LoRA 1, deploy, evaluate. Promote if better, else roll back |
| 2 | LoRA 1, if promoted | Generate at the new frontier, curate, fine-tune LoRA 2, deploy, evaluate, promote or roll back |

- Training examples are chat JSONL: `system`, `user` (scene), `assistant` (the winning plan). Confirm the exact format Crusoe's fine-tuning expects.
- Aim for at least 500 examples per round. If self-generated successes fall short, the curator tops up with teacher solutions from Super or Ultra that pass the verifier, and the ledger labels them as teacher data.

### Path B: in-context recursion (fallback, and runs alongside A)

- Each round adds verified solutions to `library/v<n>`.
- For each scene, the solver gets the 3 nearest library examples in its prompt. Nearness is computed on hand-made scene features (block count, size and mass stats, goal type), so no embedding model is needed.
- A library version goes through the same evaluate, promote and rollback steps as an adapter.
- A round takes minutes instead of waiting on a fine-tune job.
- **MemoryAlloy hook:** put the fixed instructions and a fixed core set of examples first, so every solver call shares that prefix, and put the retrieved examples after it.

**Choosing the path:** at 12:45 PM, submit a tiny dummy fine-tune job (50 examples) only to time submit, train and deploy end to end.
- **60 minutes or less:** Path A is the headline, and Path B runs alongside it.
- **More than 60 minutes:** you get one LoRA round at most. Path B becomes the headline, and the single LoRA result is a bonus.

## Recovery behaviors (the "walk away" proof)

| Failure | Detection | Response | Shown in demo |
|---|---|---|---|
| Rate limit (429) or a timeout | HTTP status, 60 s timeout | Exponential backoff, and cut concurrency in half | Counter |
| Solver returns malformed JSON | Parse fails | Counts as a fail. During data generation only, one retry that includes the parse error | Counter |
| Solver worker hangs | No heartbeat for 30 s | Kill and respawn, then requeue the scene | Event feed |
| Simulator process crashes | Pool worker dies | Respawn the pool and requeue the scene | Event feed |
| Fine-tune job fails | Job status | Retry with half the learning rate or half the data, at most 2 times. Then run a Path B round | Event feed |
| Deployment fails or stalls | No ready status within 20 min | Keep the champion and retry next round | Event feed |
| New model regresses | Held-out pass@1 down more than 2 points | **Roll back** to the previous champion, raise the curator's minimum score, and log why | **Yes, front and center** |
| Orchestrator crashes | Process exits | `run.py --resume` restarts from `state.json` | **Yes: `kill -9` it live and watch it resume** |
| Budget runs out | Per-agent token counters | Stop data generation, then finish eval and report | Counter |

If no regression happens naturally, deliberately force one, for example a learning rate 10x too high on one fine-tune, so the rollback appears in the ledger. Say in the demo that it was injected.

## Models

| Role | Model | Why |
|---|---|---|
| Improving solver | Nemotron 3 Nano 30B-A3B | Small, cheap and fast to sample; the sponsor's model family |
| Orchestrator, curriculum | Nemotron 3 Super 120B-A12B | Better planning, and it's called only a few times per round |
| Near-miss repair, teacher | Nemotron 3 Ultra, or DeepSeek V4 Pro | The strongest available model; called rarely |

**Must confirm at the Crusoe table by 12:30 PM:**
1. Which base models Serverless Fine-Tuning supports, and whether Nemotron 3 Nano is one of them.
2. The training data format, and the limits on job size.
3. How long a job takes from submit to deployed adapter.
4. Rate limits per key, and whether hackathon credits cover fine-tuning.

**If Nemotron can't be fine-tuned on Crusoe:** fine-tune the smallest supported model as the solver, keep Nemotron in every other role, and say so plainly. Path B still lets Nemotron Nano be the improving solver.

## Budget

These are estimates. I didn't check Crusoe's prices.

| Item | Per round | 3 rounds + evals |
|---|---|---|
| Solver calls | 300 scenes × 8 = 2,400 calls, about 1,200 tokens each, about 3M tokens | About 10M tokens |
| Held-out evals | 250 calls | About 1,000 calls |
| Orchestrator, curriculum, repair | Under 100 calls | Under 300 calls |
| Fine-tune jobs | 1 | 2 to 3, at unknown cost; ask Crusoe about credits |

## Repo layout

```
keystone/
  run.py                  # python run.py --goal "..." [--resume]
  sim/scenes.py           # scene generator, tiers, seeds
  sim/verify.py           # PyBullet verifier, fail types, GIF renderer
  agents/llm.py           # Crusoe client: retries, timeouts, token accounting
  agents/orchestrator.py
  agents/curriculum.py
  agents/solver.py
  agents/curator.py
  agents/trainer.py       # fine-tune + deploy wrapper
  eval/heldout.py
  ledger/                 # events.jsonl, state.json, rounds/, library/
  dashboard/app.py        # Streamlit
```

## Timeline and owners (4 people)

| Time | P1: infra and training | P2: simulator | P3: agents | P4: eval and demo |
|---|---|---|---|---|
| 12:15 | Crusoe key, `llm.py`, `/models` list | `scenes.py`, T1 and T2 | Orchestrator skeleton, ledger, `state.json` | Confirm the deadline and submission form, then the held-out seed set |
| 12:30 | Crusoe table: the 4 questions | `verify.py` with fail types | Solver pool with heartbeats | `heldout.py` |
| 12:45 | **Submit the dummy fine-tune (timing probe)** | Process pool, then T3 overhang | Curriculum and curator | Dashboard skeleton |
| 1:30 | **Gate:** base Nano scored on 250 held-out scenes | GIF renderer | Wire `collect`, `curate`, `evaluate` | Base pass@1 per tier on the dashboard |
| 2:00 | Round 0 data generated | Help P3 | Path B round 1 running | Event feed live |
| 2:15 | **LoRA round 1 submitted.** Then hands off | Hands off the loop | **Hands off the loop** | Start the "unattended since" clock |
| 2:15 to 4:00 | The loop runs alone. Nobody touches `run.py` | Base vs current GIFs on 3 demo scenes | Write the README | Rehearse the demo, record a backup video |
| 3:15 | (agent) LoRA 1 deployed, evaluated, promoted or rolled back | | | |
| 3:30 | (agent) Round 2 starts if turnaround allows | | | |
| 4:00 | **Feature freeze** | Final GIFs | Final numbers into the README | Final video |
| 4:30 | Submission text | | | Upload video |
| 4:45 | **Submit** | | | |

**Gates:**
- **12:45 PM:** the dummy fine-tune is submitted. If it can't be, Path B is the headline from now on.
- **1:30 PM:** the verifier works and base Nano is scored. If base success on T1 is under 5%, lower difficulty and seed with teacher solutions. If it's over 90%, shift the mix to T2 and T3.
- **2:15 PM:** some recursion loop is running unattended, whether Path A or Path B.
- **3:30 PM:** at least one round of measured change on the held-out set, up or rolled back. If neither happened, demo Path B's curve.

## Dashboard

A Streamlit app that reads `ledger/`.

1. **Curve:** held-out pass@1 per round, a line per tier.
2. **Agents:** each agent with status (running, idle, restarted) and token count.
3. **Event feed:** orchestrator decisions with their rationale, rollbacks highlighted.
4. **Before/after:** base vs champion GIFs on the same 3 scenes.
5. **Counters:** attempts verified, tokens used, "human interventions: 0", time unattended.

## Demo script (3 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00 | GIF of base Nemotron's tower collapsing | "Language models don't get physics. There's no labeled data for this, and we didn't make any." |
| 0:20 | The agent diagram | "We gave a team of Nemotron agents one goal and walked away at 2:15." |
| 0:45 | Dashboard: curve, event feed | "It wrote its own curriculum, graded 7,000 attempts in a simulator, fine-tuned on Crusoe, deployed, and tested itself." (Use the real numbers.) |
| 1:30 | The rollback line in the feed | "Here it shipped a worse model, caught it on the held-out set, and rolled back. No human." |
| 1:50 | `kill -9` the orchestrator, then `--resume` | "Kill it, and it picks up where it left off." |
| 2:15 | Base vs champion GIFs side by side, and the pass@1 table per tier | "Same scenes. Base: X%. Now: Y%. On 7-block towers it never trained on: Z%." |
| 2:45 | Crusoe and Nemotron usage line | "Every agent ran on Nemotron through Crusoe Managed Inference. Fine-tuning and deployment ran there too." |

## Judge questions to prepare

- **"Why not just use a physics solver?"** The point is teaching a language model physical reasoning without labels. The simulator is the grader, not the policy.
- **"Isn't this just rejection sampling, or expert iteration?"** The training signal is, on purpose, because it's proven. What's new is that an agent team runs the whole loop, curriculum, deploy and rollback included, with no human and no labels.
- **"Is it really self-improvement, or distillation from Ultra?"** Show the honesty column: the share of self-generated data per round.
- **"Does it generalize?"** Show the held-out tiers and the Unseen 7-block set.
- **"What decides things, the LLM or your code?"** The LLM picks actions and the code enforces safety rules. Every decision and its rationale is in the event feed.
- **"What happens if you leave it for a day?"** It keeps running rounds until it hits the goal, plateaus for 2 rounds (then it stops itself), or runs out of budget.

## What we claim and don't claim

- **Claim:** measured held-out improvement on these stacking tiers, produced by an unattended agent loop.
- **Don't claim:** general physical understanding, sim-to-real transfer, or results beyond the tiers tested.

## Stretch, only after 4:00 PM if everything works

- A pick-and-place variant with a tabletop arm in MuJoCo.
- A transfer test: train on height goals only, evaluate on overhang.

## Sources

- [Event page](https://cerebralvalley.ai/e/crusoe-recursive-agents-hackathon) and its [markdown version](https://cerebralvalley.ai/e/crusoe-recursive-agents-hackathon.md): opened. The deadline fields come from the page data.
- [Crusoe Managed Inference product page](https://www.crusoe.ai/cloud/managed-inference): opened. Model hub, MemoryAlloy, Serverless Fine-Tuning and Self-Serve Deployments.
- [Crusoe Serverless Inference docs](https://docs.crusoecloud.com/serverless-inference/index.html): search result only.

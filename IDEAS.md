# Recursive Agents Hackathon: Top 5 Ideas

As of Sat Oct 17, 2026, 12:05 PM PDT. Sources are at the end.

## Read this first: the deadline is probably 5:00 PM

- The event page's data has `hackathonSubmissionDueDate: "2026-10-18 00:00:00"` and `submissionDueDateGracePeriodHours: 1`. Its markdown version says "submissions due Oct 18, 2026 UTC".
- 00:00 UTC on Oct 18 is **5:00 PM PDT today**, with grace to 6:00 PM. The event runs until 10 PM, so 5 to 10 PM is likely demos and judging.
- That leaves about **5 hours** of build time from noon, not 9 to 10. Confirm with an organizer now. Every clock time below assumes 5 PM.
- The details page does load. It has rules and a waiver. I found no judging rubric, prize split or schedule in it. Entrants keep their IP, and organizers get a non-exclusive license to the submission.
- The theme, in the page's words: "Build a team of agents you can walk away from." They should plan, spin up sub-agents, delegate, communicate, check their own progress, recover when things break, and keep going. Required: Crusoe Managed Inference and the Nemotron family.

## Ranking

Scores are 1 to 5 and are my judgment. No rubric is published, so "theme fit" and "sponsor fit" stand in for one.

| # | Idea | Feasible by 5 PM | Theme fit | Sponsor fit | Novelty | Your skills | Total |
|---|---|---|---|---|---|---|---|
| 1 | Recursive Nemotron: physics self-improvement | 3 | 5 | 5 | 4 | 4 | **21** |
| 2 | Night Shift: self-healing training supervisor | 5 | 5 | 3 | 4 | 4 | **21** |
| 3 | Grid-Following Compute | 4 | 3 | 4 | 3 | 5 | **19** |
| 4 | Divergence: PyTorch bug-hunting swarm | 4 | 4 | 3 | 4 | 3 | **18** |
| 5 | Red/Blue Loop: self-hardening agent app | 4 | 4 | 4 | 3 | 2 | **17** |

#1 and #2 tie. #1 ranks first because the team is already on it and it uses Crusoe's fine-tuning and deployment, not just inference.

**Recommendation:** keep #1, and use #2's supervisor (verify each fix, auto-revert) as its recovery story. If gate 2 under #1 fails at 1:30 PM, switch #1 to in-context recursion (described below) rather than changing ideas.

## Shared stack (all five)

- **Endpoint:** OpenAI-compatible at `https://api.inference.crusoecloud.com/v1/`. Check `/models` for exact model IDs.
- **Models seen on Crusoe's model hub and docs:**
    - Nemotron 3 Nano 30B-A3B, Super 120B-A12B and Ultra.
    - DeepSeek V4 Pro, with a 1,048,576-token context.
    - Gemma, GLM (Z.ai), gpt-oss, Llama and Qwen families.
- **MemoryAlloy:** a cluster-wide KV cache. Crusoe claims up to 9.9x faster time to first token than vLLM on Llama-3.3-70B.
    - Pitch hook: give every sub-agent the same long prefix (task spec, memory ledger) and vary only the tail, so fan-out gets cheap.
- **Fine-tuning:** Serverless Fine-Tuning and Self-Serve Deployments are listed on the product page. Time per job is unknown, so ask at the Crusoe table.
- **Pattern every idea uses:**
    - Super or Ultra as the planner.
    - Nano as cheap workers.
    - A non-LLM verifier that decides success.
    - A JSONL ledger as memory.
    - Heartbeats and per-agent token caps, so the orchestrator can restart dead workers.

## 1. Recursive Nemotron: physics self-improvement (current direction)

- **Problem:** LLMs are weak at intuitive physics. They place blocks in towers that fall, and there's no labeled data for stable placement under random block sizes and masses.
- **Pitch:** an agent team turns base Nemotron Nano into a better block stacker with no human labels. Each round's fine-tuned model replaces the solver workers that generate the next round's data.
- **Task:** stable stacking.
    - The model places the next irregular block as (x, y, yaw).
    - Success: the tower stands for 3 s in PyBullet and hits a target height or overhang.
    - Difficulty knob: block count, variance in size and mass, and the overhang target.
- **Agent team:**
    - **Orchestrator (Super):** owns the goal "raise held-out success from X% to Y%" and decides when to raise difficulty or stop.
    - **Curriculum agent:** keeps scenes where the current model succeeds 30 to 70% of the time.
    - **Solver workers:** the current Nano or the latest adapter, making N attempts per scene in parallel.
    - **Verifier:** headless PyBullet in a process pool. Malformed JSON, overlapping blocks and out-of-bounds placements all count as failures.
    - **Curator:**
        - Keeps the best-of-N successes.
        - Has Ultra repair near misses, and keeps a repair only if the simulator passes it.
        - Holds out a fixed seed set that never enters training.
    - **Trainer:**
        - Submits the LoRA job, polls it, and retries with a lower learning rate or less data on failure.
        - Deploys the adapter and runs the eval gate.
        - **Rolls back to the previous adapter on a regression.** Show this on screen.
    - **Ledger:** per round, the data size, adapter ID, held-out score and failure taxonomy.
- **Fallback if fine-tuning is slow (in-context recursion):**
    - Each round adds simulator-verified solutions to a retrieval library, and solvers get the k nearest verified examples in their prompt.
    - The loop, ledger and rollback stay the same: drop a library version if the held-out score falls.
    - A round takes minutes, not an hour. Run it while the one LoRA job trains, and show both curves.
- **Demo moment:**
    - A split screen of base vs latest on the same held-out scenes, with towers falling vs standing.
    - A success-per-round curve.
    - A ledger line showing one rollback.
- **Pitch metric:** held-out pass@1 per round, by difficulty tier.
- **Novelty:** STaR, ReST-EM and expert iteration self-train against a verifier on text tasks. The new part is a physics simulator as the verifier and an agent team that owns the whole loop, deploy and rollback included. I didn't run a fresh search.
- **Gates:**
    1. **12:30 PM:** one Crusoe call works. At the Crusoe table, ask which base models Serverless Fine-Tuning supports and how long a job takes from submit to deployed.
    2. **1:30 PM:** the verifier works and base Nano is scored on 200 held-out scenes. If base success is under 5%, seed with Super or Ultra solutions that pass the simulator, and lower the difficulty.
    3. **2:15 PM:** round 1 LoRA is submitted. If submit to deployed takes over 60 minutes you get one LoRA round, so make in-context recursion the main loop.
    4. **4:00 PM:** feature freeze.
- **Risks and guards:**
    - **Slow fine-tune turnaround:** fall back to in-context recursion.
    - **Long `<think>` traces in the training data:** train on a short rationale plus the answer, and cap tokens at eval.
    - **Gains too small to see:** report by difficulty tier, since the middle tier moves first.

## 2. Night Shift: self-healing PyTorch training supervisor

- **Problem:** training runs die at 3 AM from NaNs, learning-rate blowups, OOMs and bad shards, and a person restarts them by hand.
- **Pitch:** give it a training goal and walk away. The agent team watches, diagnoses, fixes, and checks its own fix.
- **Agent team:**
    - **Orchestrator (Super):** owns the run.
    - **Watchers (Nano):** poll a metrics JSON every 30 s.
    - **Diagnoser:** gets the logs, the loss curve and the incident memory.
    - **Fixer:** limited to typed actions: roll back to a checkpoint, scale the LR, skip a shard, cut the batch, restart.
    - **Verifier:** watches K steps after each fix and auto-reverts if it fails.
    - **Chaos agent:** injects faults on a schedule.
- **Training task:** a tiny GPT on TinyShakespeare on CPU or MPS, so incidents happen within minutes.
- **Demo moment:** a judge clicks "inject fault" and watches it recover. A baseline run with the same faults and no supervisor dies.
- **Pitch metric:** % of faults recovered with no human, and mean time to recovery, across 6 fault types.
- **Novelty:** W&B and Neptune alerts are passive, and restarts on large runs are manual or rule-based auto-resume. The new part is diagnosis plus typed repair plus verify-and-revert. I didn't run a fresh search.
- **Merging with #1:** Night Shift's Fixer and Verifier are #1's Trainer rollback.
- **Gates:**
    1. **1:30 PM:** one fault recovers end to end. A hardcoded fix is fine.
    2. **4:00 PM:** at least 4 of 6 fault types recover unattended in a 20-minute run. Demo only the ones that do.
- **Risks and guards:**
    - **The LLM makes things worse:** typed actions only, plus verify-and-revert.
    - **Training is too slow to show incidents:** use a tiny model.
    - **A sub-agent hangs or burns budget:** heartbeats and token caps, and the orchestrator restarts it. That restart is another recovery to show.

## 3. Grid-Following Compute: agents that schedule training around power prices

- **Problem:** training and fine-tuning jobs run when they're submitted, whatever the grid price or carbon intensity.
- **Pitch:** agents take job requests in plain language and schedule checkpointable PyTorch jobs against price or carbon forecasts. They pause and resume jobs, and replan when a forecast misses or a job fails. Crusoe's founding story is energy-first compute, so this lands with their judges.
- **Agent team:**
    - **Intake agent:** parses each request into a deadline, GPU-hours and a priority.
    - **Solver:** a greedy or MILP solver makes the schedule. The LLM doesn't do the math.
    - **Monitor:** detects forecast misses and job failures.
    - **Replanner:** adjusts the schedule.
    - **Reporter:** narrates the savings.
- **Data:** replay a recorded day of grid prices from a CSV at 100x speed.
- **Demo moment:** a timeline with $ and kgCO2 counters against a FIFO baseline. Inject a forecast miss and a job crash mid-replay.
- **Pitch metric:** % of cost or CO2 saved vs FIFO, with every deadline met.
- **Fit to you:** extends your energy dispatch agent work.
- **Novelty:** Google's carbon-intelligent computing shifts flexible load, and CodeCarbon measures emissions. The new part is plain-language intake plus agentic replanning and recovery. I didn't run a fresh search.
- **Gates:**
    1. **1:30 PM:** price data loaded and replaying.
    2. **3:30 PM:** the scheduler beats FIFO in simulation.
- **Risks and guards:**
    - **"Why not just a solver?":** the solver does the math and the agents handle intake, failures and replanning. Say this in the pitch.
    - **Live data needs accounts:** replay a CSV.
    - **The demo feels abstract:** show the timeline and counters.

## 4. Divergence: agent swarm that hunts PyTorch numerical bugs

- **Problem:** PyTorch backends (MPS, torch.compile with Inductor) sometimes return different numbers than eager CPU. People find these bugs by hand or with research fuzzers.
- **Pitch:** walk away, and come back to minimized, deduplicated, confirmed divergences with draft GitHub issues. The audience is the PyTorch Conference.
- **Agent team:**
    - **Planner (Super):** picks op families and dtypes, and moves workers toward families that produce hits.
    - **Generators (Nano):** write small PyTorch programs.
    - **Executor:** runs each program on eager CPU vs MPS, and vs Inductor on Linux.
    - **Skeptic:** re-runs in float64 with fixed seeds and per-dtype tolerances to remove false positives.
    - **Minimizer:** shrinks each repro.
    - **Deduper:** checks hits against known issues.
    - **Reporter:** drafts the GitHub issue.
- **Checked on this Mac at 11:58 AM:**
    - torch.compile with Inductor fails on both torch 2.6.0 and 2.14.1, with `Library not loaded: @rpath/libc++.1.dylib`.
    - MPS works: CPU vs MPS max diff was 2.4e-7 on a test function. `backend="aot_eager"` also works.
    - So run MPS vs CPU locally. Inductor needs a Linux box.
- **Demo moment:** a live counter of programs tried, candidate divergences, divergences confirmed by the skeptic, and unique ones after dedupe. Show one minimized repro.
- **Pitch metric:** confirmed unique divergences per hour, and the false-positive rate before vs after the skeptic.
- **Novelty:** TitanFuzz and FuzzGPT (LLM fuzzing of deep learning libraries) and WhiteFox (LLM compiler fuzzing) are research systems. The new part is a persistent agent team that also triages and reports. I didn't run a fresh search.
- **Gates:**
    1. **1:00 PM:** one CPU-vs-MPS differential test runs inside the loop.
    2. **3:30 PM:** at least one divergence survives the skeptic. If none does, point the swarm at an older torch version with known bugs and report how many it rediscovers.
- **Risks and guards:**
    - **Every hit is float noise:** the skeptic plus per-dtype tolerances.
    - **Every hit is already known:** dedupe them, and they still count for the rediscovery demo.
    - **No hits in 5 hours:** seed with known bugs.

## 5. Red/Blue Loop: agents that attack and harden an agent app

- **Problem:** an LLM agent app's guardrails get tested once, by hand, and then drift.
- **Pitch:** red agents attack a target app all day, blue agents patch its guardrails, and a referee re-runs every past attack after each patch. Attack success rate falls on a live chart while nobody touches it.
- **Agent team:**
    - **Orchestrator:** runs the rounds.
    - **Red team (Nano):** mutates attacks (prompt injection, jailbreaks, data exfiltration through tools), seeded from garak probes.
    - **Target:** a small tool-using agent with a planted secret.
    - **Blue (Super):** edits the NeMo Guardrails config and the system prompt.
    - **Referee:** scores a held-out attack set, a regression suite of past attacks, and a set of benign tasks. The benign tasks stop blue from "fixing" things by refusing everything.
- **Sponsor angle:** garak and NeMo Guardrails are NVIDIA's open-source tools, and a Nemotron model is the judge.
- **Demo moment:** the attack-success curve falls over rounds while benign success stays flat. Show one attack that worked in round 1 and fails in round 5.
- **Pitch metric:** held-out attack success rate vs benign task success.
- **Novelty:** PyRIT, promptfoo and garak scan for weaknesses, and MART (Meta, 2023) does multi-round automated red-teaming for training. The new part is a closed loop that patches a deployed app's guardrails and regression-tests them. I didn't run a fresh search.
- **Gates:**
    1. **1:30 PM:** the target, 20 seed attacks and referee scoring all work.
    2. **3:30 PM:** three full red/blue rounds run unattended.
- **Risks and guards:**
    - **Blue overfits by refusing:** the benign set catches it.
    - **Installing garak or NeMo Guardrails eats an hour:** time-box it to 30 minutes, then fall back to hand-written probes and a prompt-level guard.
    - **The demo looks like scrolling text:** one chart and one before/after.

## Cut

- **Gap Hunter** (adversarial sim-to-real with PPO): CPU PPO plus an LLM loop doesn't fit in 5 hours, and it reuses #1's simulator for less payoff.
- **Generic deep-research or coding-backlog agents:** Crusoe already published its own Nemotron 3 Ultra + LangChain Deep Agents demo, and Devin, Codex and OpenHands already cover coding.
- **Kernel Foundry** (Triton kernels): needs an NVIDIA GPU, and Sakana's AI CUDA Engineer and KernelBench already cover it.
- **Paper Repro Crew:** PaperBench and Paper2Code already cover it, and 5 hours is too short for a meaningful reproduction.

## Next 30 minutes (any pick)

- **Person 1:** get the Crusoe key, make one call each to Nano and Super through the OpenAI client, and save the `/models` list.
- **Person 2:** build the verifier or environment (the PyBullet scene, training run, fuzz executor, target app or price replay).
- **Person 3:** write the orchestrator loop, the JSONL ledger and the heartbeats.
- **Person 4:** confirm the deadline with organizers, ask Crusoe about fine-tune turnaround, and start the demo screen.

Submit by 4:45 PM. The grace hour is for emergencies.

## Sources

- [Event page](https://cerebralvalley.ai/e/crusoe-recursive-agents-hackathon): opened. The deadline fields come from the page data.
- [Event page, markdown version](https://cerebralvalley.ai/e/crusoe-recursive-agents-hackathon.md): opened.
- [Crusoe Managed Inference product page](https://www.crusoe.ai/cloud/managed-inference): opened.
- [Strands Agents, Crusoe provider (base URL)](https://strandsagents.com/docs/integrations/model-providers/crusoe/): search result only.
- [Crusoe blog: Nemotron 3 Ultra + LangChain Deep Agents](https://www.crusoe.ai/resources/blog/nvidia-nemotron-3-ultra-langchain-deep-agents-on-crusoe-managed-inference): search result only.
- [Crusoe Serverless Inference docs](https://docs.crusoecloud.com/serverless-inference/index.html): search result only.

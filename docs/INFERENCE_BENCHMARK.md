# Inference benchmark (layer 4)

**What it is.** Serving tokens is the top layer of an AI data center, and it is what Crusoe sells. The question is the cheapest way to call Nemotron on Crusoe Managed Inference that still gets every answer right. The workload is a fixed set of data-center operations questions built from Rikin's DoneTwin halls, so the facts are the same ones the hall-layout layer optimizes. Each task is one hall with 12 questions: 4 types, 3 questions each. Splits are 4 train, 3 dev and 4 held-out halls.

**The 4 question types**, with examples from seed 7:

| Type | Example | Answer | Tolerance |
|---|---|---|---|
| cdu_headroom | CDU_1 is rated 750 kW and cools R2 (GB300-NVL72), R5 (GB200-NVL72) and R6 (GB300-NVL72); other racks are distractors. How many kW of cooling headroom are left? | 334 kW | ±0.5 kW |
| busway_left | Row A is fed by two 750 kW busways at 0.8 derate and holds 1 GB300-NVL72 and 2 HGX-B200. How many kW of 2N power budget are left? | 428 kW | ±0.5 kW |
| hose_slack | Given a CDU port position, its hose length and a rack center, how much hose is left on a Manhattan run? | 7.46 m | ±0.05 m |
| rack_fit | How many more GB200-NVL72 racks fit in row A's 2N budget? | 3 | exact |

**How it is checked.** Code computes every exact answer from the hall data. A reply counts as correct only if it is a finite number within the type's tolerance. No LLM judges anything. A hall passes at 90% accuracy or higher. Cost is Crusoe's billed tokens times the published price, and latency is recorded too.

**Baseline (v0), deliberately expensive.** Nemotron Super for every type, reasoning on, max_tokens 4000, one question per call.

**What the team can change** (the Reviewer proposes, the gate decides):
- Model per question type: Super, Nano or Lightning.
- Reasoning on or off, per type.
- max_tokens, overall and per type (256 to 32,768).
- Batch size: 1 to 12 questions of one type per call.
- Temperature, plus prompts and playbooks per type.

**The trade-off.** On one live call, reasoning on used about 25x the tokens of reasoning off (180 vs 7). Cheaper settings can fail in 3 ways, and the failure is logged by type:
- a wrong answer (`wrong:<type>`);
- a reply cut off at the token cap (`truncated:<type>`);
- invalid JSON (`bad_json:<type>`). With batching, one bad reply loses the whole batch.

**How a change is accepted.** The candidate and the current version answer the same 3 dev halls. A change is kept if it cuts mean cost per hall by 15% or more with no drop in accuracy and no lost passes (`MIN_COST_CUT = 0.15` in `team/evolve.py`), or if it raises accuracy by 0.05 or more. The final report runs on the 4 held-out halls, which the team never saw.

**Result (held-out, 4 halls, 48 questions).** v0 got 47 of 48 held-out answers right at $0.00109 per correct answer; the final version got 48 of 48 at $0.00075, 31% less per correct answer (30% less per hall). The accepted change was one paper: batch 3 questions per call (claimed −55% cost, measured −31.8% on dev, verdict reproduced). Calls per hall fell from 12 to 4 and hall wall time from 163 s to 77 s, but median call latency rose from 9.7 s to 19.0 s. Source: `reports/inference_opt_2/`. One extra correct answer is within noise, so the claim is the cost cut, not an accuracy gain.

**Limits.**
- This is our own workload, not a public benchmark, so there are no outside records to compare against.
- It tunes how we call the model (routing, reasoning, token budgets, batching), not Crusoe's serving stack (GPU scheduling, KV cache, quantization), which we cannot touch.
- Small sample: 4 held-out halls, 48 questions.

Code: `benchmarks/inference_opt/` (`workload.py`, `team_runner.py`, `verifier_api.py` on :9004), `scripts/run_inference_opt.py`, `tests/test_inference_opt.py`.

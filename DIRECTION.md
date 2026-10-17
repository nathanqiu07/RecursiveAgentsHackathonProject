# One Direction: DoneTwin, Hardened by a Loophole Agent, Scored on Done-Bench

As of Sat Oct 17, 2026, about 1:45 PM PDT. Merges the two branches: `main` (Keystone research, SA's verified leaderboard) and `spec/donetwin` (Rikin's DoneTwin spec). Proposed, not yet agreed by the team.

## Decision

**Build on DoneTwin. Add one new agent, the Loophole agent, as the key novelty. Prove it on Done-Bench, a hidden-verifier benchmark hosted on SA's live dashboard.**

**Pitch:** "Agents say done against a definition of done they wrote themselves. Who checks the definition? Our Loophole agent attacks it: it builds layouts that pass every check yet are measurably wrong. Done means the definition survived the attack, and an independent hidden verifier agrees."

## Why this resolution

| Perspective | What it brings | What it drops |
|---|---|---|
| **DoneTwin** (Rikin, `spec/donetwin`) | The base. Working loop, tested offline with a scripted LLM: Spec, Layout, sandbox, geometry verifier, vision inspector, supervisor, USD export, 3D twin UI. A domain Crusoe lives in: GB200 NVL72 data halls | Nothing. It needs a live Crusoe run (waiting on the API key) |
| **Keystone** (Nathan, `main`) | Its evaluation discipline: a frozen held-out set, a non-LLM verifier as ground truth, and a promote or roll back gate. Its recursion: the system improves across rounds | Fine-tuning and the overhang target. Crusoe can't fine-tune Nano, and jobs "run for hours" (see `NOVELTY.md`). There is no Keystone code yet |
| **Verified leaderboard** (SA, `dashboard/`, `hosting/`) | Independent scoring. The hosted copy is live at https://crusoe-benchmark-dashboard.shardul711.chatgpt.site with zero benchmarks. Its rule, "only verified scores rank", is DoneTwin's thesis | The idea of agents solving arbitrary third-party benchmarks |

The three share one thread: **a measured verifier, not an LLM, decides success.** The merged project pushes that thread one level up. The checks themselves get verified.

## The gap this closes

DoneTwin's own spec names it (section 17, "Spec gaps"): if the Spec agent forgets a `zone_clear` for a keep-out, a layout is "verified" while racks sit on the cable vault. Every self-verifying agent has this hole. The verifier is only as good as the definition of done, and the same agent wrote both.

## Key novelty: the Loophole agent

**Role:** a contractor paid to pass inspection with the least work. It reads the Definition of Done and writes a layout program that passes every check while breaking a requirement the spec forgot.

**Model:** Nemotron 3 Nano 30B-A3B by default, with 4 attempts in parallel per round. A cheap adversary keeps the expensive Super spec-writer honest. Fall back to Super if Nano's code fails too often.

### Phase order

The Loophole phase runs **before** the layout loop, so the definition of done is fixed before any work counts. No moving goalposts mid-run.

```
goal -> Research -> Spec agent -> Definition of Done v1
                                      |
             +------------------------+
             v
   LOOPHOLE ROUND (4 Nano attempts in parallel)
     each returns: claim, witness check, layout program
             |
             v
   sandbox -> verifier: does the layout pass ALL of DoD vN?        (must be yes)
                        does the witness check FAIL, measured?     (must be yes)
                        is the witness already in the DoD?         (must be no)
             |
     confirmed loophole -> Spec agent patches -> DoD vN+1 -> next round
     no confirmed loophole for 2 rounds, or 4 rounds used -> DoD frozen
             |
             v
   existing DoneTwin layout loop against the hardened DoD
             |
             v
   DONE.md: "Definition of Done survived N loophole attempts; M gaps patched", with each counterexample
```

### Rules that keep it honest

- **Every loophole is a measured counterexample.** The witness is a check from the existing library, with parameters, and the deterministic verifier must measure it failing. An LLM opinion alone never patches the spec.
- **The Spec agent may reject a loophole** only with a written reason ("not required by the goal"). Rejections are listed in DONE.md for a human.
- **A patch must still admit a valid layout.** Each scenario keeps one reference layout known to be correct: the generator writes one per variant, and `sorting_cell` uses the mock run's verified 10/10 layout. A patch that rejects the reference is refused. Without this rule, a spec that rejects everything would "survive" every attack. The idea comes from the hacker-fixer loop's solver check.
- **The loophole layout starts from the base scene,** like every layout program, and runs in the same sandbox.

### Output and UI

- **Event types:** `loophole {round, attempt, claim, witness, dod_pass, witness_fail, verdict, scene}` and `spec_patch {added, why, source}`.
- **Twin rendering:** the twin renders the loophole scene in red, with a banner such as "Passes all 14 checks. 6 racks on a 1,200 kg/m2 slab."

### Examples the existing scenarios already contain

| Scenario | Spec gap | Loophole layout | Witness check |
|---|---|---|---|
| `data_hall` | No `zone_clear` on the slab vault | 4 racks on BW_D over the vault (about 1,890 kg/m2 on a 1,200 kg/m2 slab) | `zone_clear(slab_vault)` |
| `data_hall` | No `aisle_width` | Rows 0.3 m apart | `aisle_width(min_m=1.2)` |
| `data_hall` | `capacity` without `redundancy` | All 16 racks on 3 CDUs | `redundancy(cdu)` |
| `sorting_cell` | `zone_clear(walkway)` without `include_envelopes` | Robot base at y = 3.9, so the 3.1 m reach covers the walkway | `zone_clear(walkway, include_envelopes=true)` |

## Proof: Done-Bench

**Question it answers:** how often does an agent say done when it isn't?

**Setup:**
- **Variants:** a generator perturbs `data_hall` and `sorting_cell`: column positions, keep-out and vault positions, target MW, busway derate, CDU count, hose lengths and the walkway. **20 variants, 10 dev and 10 held-out. Freeze the held-out set before any tuning.**
- **Hidden verifier:** for each variant, the generator also writes the full ground-truth check list with standard thresholds: every keep-out, aisle ≥ 1.2 m, NEC working space, capacity with derate, N+1, hose runs and egress. Agents never see it.
- **Hosting:** the hidden verifier is an HTTP endpoint wrapping `checks.verify`, registered as a benchmark on SA's hosted dashboard. Every run submits its final scene there, so every number on the slide is independently verified.

**Configurations:**

| Config | What it is |
|---|---|
| A. One-shot | The Layout agent runs once and declares done |
| B. DoneTwin | As built today |
| C. DoneTwin + Loophole | The key novelty |
| D. C + promoted library | The recursion round below |

**Metrics, all on the 10 held-out variants:**

| Metric | Definition |
|---|---|
| **False-done rate** (headline) | Claimed done, but the hidden verifier fails |
| Verified-done rate | Claimed done, and the hidden verifier passes |
| Escalation rate | The counter-metric: a spec that asks for everything escalates instead of finishing |
| Cost | USD per run |
| Wall time | Per run |

**Budget:** 10 variants × 4 configs × about 15 calls is about 600 calls. That is 20 minutes at 30 requests/min and about 1 minute at 600. Cost is under $5 at Super prices.

## Recursion: the definition of done learns, not the weights

This is Keystone's promote or roll back gate, applied to text instead of LoRA weights, so it needs no fine-tuning job.

1. Every confirmed loophole becomes a candidate template in `dod_library.json`, for example "`zone_clear` on every fixed keep-out, with envelopes".
2. After the dev round, promote a template only if it lowers false-done on dev variants without raising escalations by more than 1.
3. Round 2 seeds the Spec agent with the promoted library. Report on held-out variants whether false-done falls and how many loopholes per run are still found. If held-out false-done goes up, roll back and say so.

**Claim:** "Across runs, the team learns what done means for a data hall."

## Claims

From a prior-art pass on Oct 17. The search was not exhaustive, so we claim "first" nowhere.

| Piece | Closest prior work | What we can say |
|---|---|---|
| Loophole agent | [Hacker-fixer loop](https://arxiv.org/abs/2606.08960) (June 2026): a hacker passes a verifier without solving, a fixer patches it, a solver confirms real solutions still pass. Those are hand-written benchmark verifiers, hardened offline. Related: [Specification Self-Correction](https://arxiv.org/abs/2507.18742) (no adversary, no measurement), [Self-Challenging](https://arxiv.org/abs/2506.01716), [AutoSpec](https://arxiv.org/abs/2606.24245) | "Hacker-fixer loops harden hand-written benchmark verifiers offline. We run that adversarial pattern at inference time against the agent's own self-written Definition of Done, and count a loophole only when a deterministic geometry check measures it." |
| False-done rate | "False success" ([arXiv 2606.09863](https://arxiv.org/abs/2606.09863)): 45 to 48% of failures in tau2-bench single-control domains. Also [ImpossibleBench](https://arxiv.org/abs/2510.20270) and [SWE-bench plausible-but-incorrect](https://arxiv.org/abs/2503.15223) (7.8% of patches) | "We report false-done rate, the false-success measure of arXiv 2606.09863, on procedurally generated data-hall layouts with a hidden geometry verifier." |
| Patch library across runs | Hacker-fixer already transfers patches. Also [self-evolving verifiers](https://arxiv.org/abs/2608.11340) | "Our spec patches persist across runs, and we measure whether false-done falls over rounds." Only with a real curve |
| Data-hall domain | [Phythesis](https://arxiv.org/abs/2512.10611) (Dec 2025): LLM plus physics-guided evolution for data-center scenes, aimed at energy efficiency | "Prior LLM data-center layout work optimizes energy. We target verifiable completion for GB200 liquid-cooled halls." |

### Do not say

- "first" or "novel", for any piece
- "we introduce false-done rate"
- "first LLM data-center layout agent"
- "CEGIS" without "-like"
- "provably done", "guaranteed complete", "unhackable", "zero loopholes", "self-verifying", "solves reward hacking"
- "recursive self-improvement", without the round-over-round curve

DoneTwin's spec one-liner says "stops only when the layout is provably right". Change it to "measurably right" before submission.

## Build plan

The DoneTwin code is not in this repo yet, so nothing below can start until it is pushed. The owners are a proposal.

| By | Owner | Gate |
|---|---|---|
| 1:55 | Rikin | DoneTwin code pushed to a branch (`feat/donetwin`), with `requirements.txt` and no `.env` |
| 2:30 | Rikin | First live Crusoe run on `sorting_cell` (needs the API key) |
| 3:00 | SA | Variant generator, hidden verifier HTTP endpoint, held-out set frozen |
| 3:15 | Nathan | Loophole phase: prompt, loop stage, events, DONE.md section. Confirms at least 1 loophole on `sorting_cell` live |
| 3:30 | Rikin | `data_hall` live end to end. UI shows the loophole scene in red |
| 3:30 | SA | Done-Bench registered on the hosted dashboard. The verifier must be publicly reachable, so it needs a tunnel |
| 4:15 | SA + Nathan | Configs A to C on held-out variants. D only if time allows |
| 4:15 | All | Code freeze |
| 4:30 | 4th teammate, or whoever is free | Record the 1-minute video. Run the replay fallback |
| 4:50 | All | Submit |

### Cut order if behind

1. Config D and the library.
2. The leaderboard hosting: show local verifier numbers and say so.
3. The `data_hall` live run: demo the loophole on `sorting_cell`, and replay `data_hall`.

**Never cut:** the held-out freeze, and the measured-witness rule.

## Blockers for the next 15 minutes

1. **DoneTwin code into the repo.** Only Rikin has it.
2. **Crusoe API key.** DoneTwin has never run live. Who has the key?
3. **Payment method** on the Crusoe account, for 600 requests/min instead of 30.
4. **Confirm with an organizer:** the "Done Means Done" track (it appears only in Rikin's spec; we have not seen it on the event page) and the 5:00 PM deadline.
5. **Dashboard limits:**
   - The verifier must be reachable from Cloudflare, so a localhost endpoint needs a tunnel.
   - Verifiers get 8 s per request.
   - Every submission needs a public GitHub commit, and the anonymous GitHub API allows about 60 requests per hour.
   - The hosted copy runs on SA's account, which only SA controls.

## 3-minute demo

| Time | Beat |
|---|---|
| 0:00 | "Agents check their work against a definition of done they wrote. Who checks the definition?" |
| 0:20 | Hand off "Deploy 2 MW of GB200 NVL72 racks in Hall B with N+1 cooling." The Spec agent writes its checks |
| 0:40 | 4 Nano loophole agents strike in parallel. Red layout: "Passes all 14 checks. Racks on the cable vault." Spec patched. Second loophole, patched. Third round finds nothing: the definition of done is frozen |
| 1:20 | The layout loop runs against the hardened definition (existing DoneTwin beats, at 4x) |
| 2:00 | Done-Bench on the leaderboard: false-done rate of one-shot vs DoneTwin vs DoneTwin + Loophole, on 10 held-out halls, independently verified |
| 2:30 | DONE.md: "survived 3 loophole attempts, patched 2 gaps". `layout.usda` in a viewer |
| 2:50 | "Done means the definition survived an attack." |

## Dropped, with reasons

- **Overhang record search:** a different product with no code and no shared demo. It stays in `NOVELTY.md` as a stretch idea.
- **LoRA fine-tuning round:** Nano is not fine-tunable on Crusoe, and job turnaround is hours.
- **Proposer/solver self-play:** reads as "Absolute Zero for blocks".

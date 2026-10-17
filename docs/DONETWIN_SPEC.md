# DoneTwin: Project Spec

As of Sat Oct 17, 2026, about 1:30 PM PDT. Track: **Done Means Done**. Stack: Crusoe Managed Inference, NVIDIA Nemotron, You.com Search, OpenUSD.

This spec describes the code as it exists now, not what we hope to build. The implementation currently lives in a separate working directory and will come to this repo in a follow-up PR. Section 13 says what is tested and what is not.

## 1. One-liner and problem

**One-liner:** DoneTwin is a team of Nemotron agents that lays out AI data halls and factory cells. It writes its own definition of done, checks the result against geometry and physics, notices when it is stuck, and stops only when the layout is provably right. If it can't get there, it asks a human.

**Problem:**

- Fitting out an AI data hall or retrofitting a factory cell takes weeks of manual iteration in spreadsheets and CAD. Rack power, busway derates, cooling loops, hose runs, columns, egress, code clearances and floor loading all interact.
- Mistakes surface late (at commissioning, or when the racks arrive), and late mistakes are expensive.
- A normal coding agent writes layout code, sees it run, says "Done!", and stops. The code ran, but the layout can still put a rack on a column, overload a busway, or put a robot's reach envelope over a walkway. Running is not the same as done.

## 2. Hackathon track fit: Done Means Done

| Track requirement | Where DoneTwin does it | Code |
|---|---|---|
| Writes its own definition of done | The **Spec agent** turns a one-line goal into explicit assumptions plus machine-checkable checks, built only from the check library. Non-negotiable invariants are then injected | `agents.spec_agent`, `agents.normalize_spec`, `checks.MANDATORY` |
| Checks with tests or independent verifiers | (1) A **deterministic geometry verifier** measures every check and returns exact correction vectors. (2) A **vision inspector** (a different model) sees only rendered drawings, never the check results. (3) **Adjudication** re-measures the inspector's claims | `checks.verify`, `agents.vision_inspector`, `loop._adjudicate` |
| Catches drifting or looping | The **supervisor** fingerprints each layout to catch loops, and tracks the penalty trend to catch drift. Either one triggers a re-plan | `scene.layout_signature`, `loop._execute` |
| Continue, re-plan, stop, or ping a human | Every iteration ends in a `decision` event: `continue`, `replan`, `done` or `escalate` | `loop._execute` |
| Proof of done | A `DONE.md` certificate listing every check with pass/fail and the assumptions made, plus a physics-ready `layout.usda` | `loop._finish`, `usd.to_usda` |

## 3. Architecture and loop

```
 one-line goal + base scene (scenarios/*.json)
        |
        v
 RESEARCH (You.com Search, optional) ---- cited code/standard snippets (NEC, NFPA, ISO 10218 ...)
        |
        v
 SPEC AGENT (Nemotron Super) ----------- Definition of Done: assumptions + checks
        |                                + injected invariants (no_collision, in_bounds, fixed_unchanged, exists)
        v
 +--> LAYOUT AGENT (Nemotron Super) ---- writes a Python program against the Scene API
 |      |
 |      v
 |    SANDBOX (subprocess | Docker | NVIDIA OpenShell) -- runs the program and returns the new scene JSON or a traceback
 |      |
 |      v
 |    GEOMETRY VERIFIER (deterministic, shapely) -------- pass/fail per check, magnitude in meters, fix vector
 |      |
 |      v
 |    RENDER (matplotlib) -------------------------------- top plan + front elevation PNG
 |      |
 |      v
 |    VISION INSPECTOR (Nemotron Omni) ------------------ independent verdict from the drawings only
 |      |
 |      v
 |    ADJUDICATION (deterministic) ---------------------- re-measures inspector claims: upheld | overruled
 |      |
 |      v
 +--- SUPERVISOR (deterministic rules) ---- continue | replan (loop/drift) | done | escalate
        |
        v
 runs/<id>/final/layout.usda + DONE.md + scene.json, streamed live to the 3D twin
```

One iteration, step by step:

1. The Layout agent gets the goal, the spec, the base scene, its last 3 attempts (code, violations with fixes, inspector concerns, positions tried) and any supervisor note.
2. The program always starts from the **base scene**, so a bad attempt can't leave the next one in a worse state.
3. The sandbox runs it with a timeout. A crash goes back to the agent as a traceback (`continue`).
4. The verifier runs every check. Penalty = sum over failing checks of `1 + sum(violation magnitudes)`.
5. The inspector runs on candidate final layouts (`vision=final`, the default), on every iteration (`every`), or never (`off`).
6. The supervisor decides what happens next (section 8).

## 4. Agents and models

All LLM calls go through one OpenAI-compatible client pointed at Crusoe Managed Inference (`https://api.inference.crusoecloud.com/v1`, overridable with `CRUSOE_BASE_URL`). It retries 5 times with exponential backoff and doesn't retry on 400/401/403/404. It strips `<think>` blocks and falls back to `reasoning_content` when a reasoning model spends its whole budget thinking. It also records tokens and USD cost per call.

| Agent | Model | Temp | Input | Output | Why this model |
|---|---|---|---|---|---|
| Research | You.com Search API (no LLM) | - | `research_queries` from the scenario | `[{query, title, url, snippet}]`, 3 per query, run in parallel | Grounds thresholds in real code text and real URLs, so the Spec agent never invents a citation |
| Spec agent | `nvidia/Nemotron-3-Super-120B-A12B` | 0.2 | Goal, domain notes, scene, catalogs, check library, references | JSON: `interpretation, assumptions, new_objects, movable, checks, escalate_if` | Needs the strongest reasoning in the loop: it has to read an ambiguous goal and pick every rule a site engineer would sign off on. It's an MoE with 12B active parameters, so it reasons well at low cost ($0.30 / $2.40 per 1M tokens in, out) |
| Layout agent | `nvidia/Nemotron-3-Super-120B-A12B` | 0.4, or 0.8 after a re-plan | Spec, base scene, Scene API doc, last 3 attempts with fixes | One Python program (< 60 lines) | Writing code under many interacting constraints is the hardest step. Super handles code and long context, and is cheap enough to call up to 6 times per run |
| Geometry verifier | none (Python + shapely) | - | Scene JSON + checks | Report with exact fixes | Ground truth must be measured, not judged |
| Vision inspector | `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` | 0.1 | Plan + elevation PNG, goal, new equipment list | `{verdict, issues[{severity, objects, description}]}` | Multimodal, so it can read drawings. It's a different model from the worker and only sees the drawings, so its errors are independent of the Layout agent's. It's also cheap ($0.30 / $1.83) |
| Supervisor | none (rules in `loop.py`) | - | Report, history, layout signatures | `decision` event | Stop, re-plan and escalate decisions must be predictable and auditable |

Model IDs can be overridden with `PLANNER_MODEL` and `VISION_MODEL`. `llm.PRICES` also lists Nemotron 3 Nano, Nemotron 3.5 Lightning and Gemma for cost accounting.

## 5. Scene model and Scene API

A scene is JSON: `floor {w, d}` in meters, plus `objects[]`. Each object has `id, kind, x, y` (the footprint center), `w` (along x), `d` (along y), `h`, `rot` (degrees), `fixed`, and optionally `z`, `points` (named interaction points relative to the object), `note`, and per-kind fields (`capacity_kw`, `derate`, `hose_m`, `tap_m`, `kw`, `kg`, `cdu`, `busway`, `reach`, `min_reach`, `model`, `reason`).

| Kind class | Kinds | Meaning |
|---|---|---|
| Physical (collide) | `pillar, conveyor, bin, robot, fence, rack, pallet, cdu, panel` | Footprints may not overlap |
| Zones (floor areas) | `walkway, keepout` | Only constrained by `zone_clear` checks |
| Overhead | `sensor, busway` | Drawn dashed, with `z` > 0 |

Catalogs (in `scene.py`):

| Model | Key specs |
|---|---|
| Robot `HD-500` | 500 kg payload, 3.1 m reach, 0.7 m min reach, 1.1 x 1.1 m base |
| Robot `MD-165` | 165 kg, 2.65 m reach, 0.9 x 0.9 m |
| Robot `LD-20` | 20 kg, 1.8 m reach, 0.6 x 0.6 m |
| Rack `GB200-NVL72` | 132 kW, 1,360 kg, 0.6 x 1.2 x 2.3 m, liquid-cooled |
| Rack `GB300-NVL72` | 142 kW, 1,400 kg, same footprint |
| Rack `HGX-B200` | 15 kW, 900 kg, air-cooled |

**Scene API** (what the agent-written program calls; `scene` is pre-bound and only `math` can be imported):

| Method | Effect |
|---|---|
| `get(id)` | Copy of an object's record |
| `ids(kind=None)` | List of ids, optionally filtered by kind |
| `has(id)` | Whether the object exists |
| `point("obj.name")` / `point("obj")` | World (x, y) of a named point, rotated with the object, or of its center |
| `add_robot(id, model, x, y, rot=0)` | Place a catalog robot (footprint and reach come from the catalog) |
| `add_rack(id, model, x, y, rot=0, cdu=None, busway=None)` | Place a catalog rack with its cooling and power feeds |
| `assign(id, cdu=None, busway=None)` | Change a rack's feeds |
| `add_object(id, kind, x, y, w, d, h, rot=0, points=None)` | Generic object (robots must use `add_robot`) |
| `move(id, x=None, y=None, rot=None)` / `translate(id, dx, dy)` | Reposition |
| `remove(id)` | Delete |
| `floor_w`, `floor_d`, `to_dict()`, `save()`, `Scene.load()` | Utilities |

Unknown ids, models or points raise `SceneError` with the list of valid options, so the agent can correct itself on the next iteration. `layout_signature(scene)` fingerprints the movable objects on a 0.1 m grid (plus rotation) for loop detection.

## 6. Check library

Every violation carries `objects`, `magnitude` (meters, or scaled kW), `detail` (what is wrong, with measurements) and `fix` (a concrete correction, usually a translation vector).

| Check | Params | What it means |
|---|---|---|
| `exists` | `id, kind` | The object is in the final layout. Catches agents that "pass" by dropping the task |
| `no_collision` | - | No two physical footprints intersect. Pairs that are both fixed in the base scene are ignored. The fix is the smallest axis move out |
| `in_bounds` | - | Every footprint lies inside the floor |
| `fixed_unchanged` | - | No fixed object was moved, rotated or removed |
| `min_clearance` | `subject, against_kinds, distance` | Gap from the subject to every object of those kinds is at least `distance` (maintenance access) |
| `reach` | `robot, point, margin` | `min_reach <= dist(base, point) <= reach x margin` |
| `swing_clear` | `robot, from, to, against_kinds, arm_width` | The swept sector from pick to place, going at least one way (CCW or CW), hits no full-height obstacle |
| `zone_clear` | `zone, include_envelopes` | No movable footprint intrudes into the zone. With `include_envelopes`, no robot's max-reach circle overlaps it either (ISO 10218-2 restricted space) |
| `aligned` | `a, b, tolerance` | Point `a` is within `tolerance` of point `b` (for example, the bin drop under the fill sensor) |
| `total_power` | `min_kw` | Sum of rack kW meets the requested deployment |
| `capacity` | `kind` (`cdu` or `busway`) | Every rack has a valid feeder, and no feeder exceeds its usable capacity. A busway's usable capacity is `capacity_kw x derate` (continuous load) |
| `redundancy` | `kind` (`cdu`) | N+1: total load fits in total capacity minus the largest unit |
| `hose_length` | `max_m` (optional) | Manhattan run from the rack center to its CDU's `port` is at most `hose_m` |
| `feed_reach` | `max_m` (optional) | The rack center is within `tap_m` of its busway's footprint |
| `aisle_width` | `min_m` | Facing racks either abut (one row) or have a gap of at least `min_m` (an aisle) |

**Mandatory invariants:** `normalize_spec` drops any check type not in the library (reported as `dropped_checks`). It always injects `no_collision`, `in_bounds` and `fixed_unchanged`, and adds an `exists` check for every new non-rack object. An agent therefore can't pass by deleting the robot, moving a column, or leaving out the hard checks. Fleets of racks are covered by `total_power` and `capacity` instead of per-rack `exists`. A malformed check, or a check that names a missing object, counts as a failure, never a crash.

## 7. Scenarios

### `data_hall` (hero): "Deploy 2 MW of GB200 NVL72 racks in Hall B with N+1 cooling."

Hall B is a 24 x 16 m floor. It has four fixed CDUs along the north wall (800 kW each, 14 m hoses, ports at y = 14.8), four overhead busways `BW_A..BW_D` at y = 3.4 / 6.2 / 9.8 / 12.6 (x 2 to 22, 800 kW with a 0.8 derate, 0.8 m tap reach), four columns, a main switchboard `PNL_1`, two egress corridors (`egress_W`, `egress_S`), a NEC 110.26 working-space keep-out in front of the panel, and a slab keep-out over a cable vault (rated 1,200 kg/m2, while a GB200 rack loads about 1,890 kg/m2). The scenario also ships 5 You.com research queries (NEC 110.26, NFPA 101 egress, aisle widths, GB200 specs, the 80% busway rule).

Traps that make a naive layout fail:

| Trap | The numbers | Naive mistake | Check that catches it |
|---|---|---|---|
| Rack count | 2,000 / 132 = 15.2, so 16 racks (2,112 kW) | 15 racks, "about 2 MW" | `total_power` |
| Busway derate | Usable capacity 640 kW per busway, so at most 4 racks (528 kW). 16 racks need **all four** busways at exactly 4 each | 8 racks on each of two busways | `capacity(busway)` |
| CDU loading and N+1 | At most 6 racks per CDU (792 kW). N+1 capacity is 2,400 kW, enough for 2,112 | Uneven assignment | `capacity(cdu)`, `redundancy` |
| Hose runs | The south row (BW_A) is 11.4 m from the CDU ports in y, leaving only 2.6 m in x. North rows have far more slack | Assigning the south row to a CDU by row index, not by x position | `hose_length` |
| Columns on busway lines | `col_C1/C2` sit on BW_B (x = 8, 16) and `col_C3/C4` on BW_D | A 2.4 m row of 4 abutting racks placed through a column | `no_collision` |
| Code and structure keep-outs | The slab vault covers x 18 to 24 under BW_C/BW_D. The NEC zone touches the east end of BW_A | Using the full busway length | `zone_clear` (the Spec agent must write these; zones are not physical, so `no_collision` misses them) |
| Rows and aisles | Racks in one row abut, and rows are at least 1.2 m apart | Spacing racks a few cm apart | `aisle_width`, `feed_reach` |

### `sorting_cell`: "Insert a heavy-duty robotic arm between conveyor line A and sorting bin B."

A 16 x 10 m cell with six columns, conveyor A (pick point at its east end, (7.2, 8.6)), a movable bin B, an overhead fill sensor S, and a pedestrian walkway along the south wall. Three ISO 10218 / OSHA research queries.

| Trap | Why a naive layout fails |
|---|---|
| "Heavy-duty" is vague | The Spec agent must make it explicit: at least 300 kg, so `HD-500` with 3.1 m reach and 90% usable |
| Midpoint placement | A robot halfway between pick and drop lands on `pillar_P5` |
| Arm swing | Between `pillar_P2` and `pillar_P5`, both sweep directions are blocked |
| Sensor alignment | Bin B starts 0.10 m from the sensor, outside the 0.05 m tolerance, so drops miss the fill sensor unless the bin moves |
| Walkway | The 3.1 m reach envelope must not cover the walkway, so the base must be at y >= 4.6 |

In the scripted mock run, the loop goes from 4 failing checks to 2, then to all 10 passing in 3 iterations, with the robot ending north-east of P5 and the bin moved under the sensor.

## 8. Supervisor policy

The supervisor is deterministic code in `loop._execute`. It is not an LLM.

| Condition | Action | Detail |
|---|---|---|
| Sandbox crash or timeout | `continue` | The traceback goes into the next prompt |
| All checks pass and no inspector concern stands | `done`, status `verified` | Either the inspector passed it, or every claim was overruled by measurement |
| All checks pass, but the inspector's concerns are upheld | `continue` | Concerns go to the Layout agent as `INSPECTOR:` lines |
| The above happens a 2nd time | `escalate` | Checks and inspector persistently disagree, so the spec is probably missing a rule. A human decides |
| **Loop:** the layout signature matches an earlier iteration | `replan` | Note: "LOOP DETECTED ... abandon this region, try the opposite side and the other sweep direction", plus every position tried. Temperature goes to 0.8 |
| **Drift:** no penalty improvement for 2 iterations | `replan` | Note: "DRIFT ... re-plan from scratch", with the best penalty so far. Temperature goes to 0.8, stall count resets |
| Otherwise | `continue` | Exact fixes go back. Temperature returns to 0.4 |
| Budget used up (`max_iters`, default 6) | `escalate` | Ships the **best** attempt (lowest penalty) with its blocking checks |

**Vision adjudication:** `minor` issues are noted only. For `major` or `critical` issues, the footprint gap is measured between every pair of objects the inspector named. If every gap is at least 0.05 m, the claim is **overruled** and the measurement is attached as evidence. Otherwise, including "not measurable", the claim is **upheld**.

## 9. Server API and event contract

`python server.py` serves `http://127.0.0.1:8000` (port from `PORT`). Each run executes in a background thread. The sandbox mode comes from `SANDBOX_MODE`, or is auto-detected in the order OpenShell, Docker, subprocess.

| Method and path | Returns |
|---|---|
| `GET /` | The 3D twin (`ui/index.html`) |
| `GET /api/scenarios` | `["data_hall", "sorting_cell"]` |
| `GET /api/scenario?name=` | Base scene JSON (404 if unknown) |
| `POST /api/run` | Body `{scenario, goal?, mock?, max_iters?=6, vision?="final"}` returns `{run_id}`. Returns 400 if `CRUSOE_API_KEY` is missing and `mock` is false |
| `GET /api/events?run=&since=N` | `{events: [...from seq N], finished}`. Live from memory, or replayed from `runs/<id>/events.jsonl` |
| `GET /api/runs` | `[{run_id, goal, status, started}]`, newest first |
| `GET /sample_events.json`, `/ui/*`, `/scenarios/*`, `/runs/*` | Static files, including run artifacts |

The UI **polls** `/api/events`; there is no SSE or WebSocket. Every event has `seq`, `t` (seconds since start) and `type`. Events are appended to `runs/<id>/events.jsonl`.

| `type` | Fields |
|---|---|
| `run_start` | `goal, scenario` (full base scene), `models {planner, vision}, sandbox` |
| `log` | `agent` (research, spec, layout, sandbox, verifier, vision, supervisor), `msg` |
| `research` | `references [{query, title, url, snippet}]` |
| `spec` | `spec {interpretation, assumptions, new_objects, movable, checks [{type, params, why, source?}], escalate_if, dropped_checks}` |
| `iteration_start` | `iter` |
| `layout` | `iter, code, note` |
| `sandbox` | `iter, ok, error, stdout, scene, points {"obj.pt": [x, y]}, mode` |
| `verify` | `iter, report {checks [{type, params, why, passed, violations [{objects, magnitude, detail, fix}]}], passed, total, all_pass, penalty}`, `geometry {robots [{id, x, y, reach, pick, place, sweep {a0, a1, radius, hits {ccw, cw}, dir}}], loads {feeder: {load_kw, cap_kw}}, links [{rack, cdu, busway}]}` |
| `vision` | `iter, verdict` (pass, fail or skipped), `issues, adjudication [{claim, measured, ruling}]` |
| `decision` | `iter, action` (continue, replan, done or escalate), `reason` |
| `done` | `status` (verified, escalated or failed), `iterations, usage {calls, prompt_tokens, completion_tokens, cost_usd}, summary, usd_path` |

## 10. 3D digital twin UI

**The UI is the window. The agents are the product.** The twin only renders the event stream above, so any client that reads the contract could replace it.

`ui/index.html` is a single file. It uses Three.js from a CDN with OrbitControls, SSAO, bloom, RoomEnvironment lighting, CSS2D labels and Line2. It shows:

- **Work order card:** a scenario picker, a goal textarea and a "Hand off to agents" button. During a run it shows "Agents working. You can walk away." with elapsed time, a **Human interventions: 0** counter and the current iteration.
- **The live scene**, rebuilt from each `sandbox` event: racks, CDUs, busways, columns, keep-outs, egress, robots with reach envelopes and arm-sweep arcs, feeder links, and violations highlighted.
- **View modes:** Thermal (T, rack exhaust heat ramp), X-ray (X, hides overhead services) and HQ post-processing (Q). Playback speed 1x, 2x or 4x.
- **KPIs:** MW online, checks passed, iterations, inference cost, agent time.
- **Side panel:** the Definition of Done with live pass/fail and cited sources, an iteration strip, the inspector verdict with adjudication, the agent log, and the layout program the agent wrote.
- **Crew strip:** Research (You.com), Spec, Layout, Sandbox, Verifier, Inspector and Supervisor, each with its model or "deterministic" badge.
- **Replay** of past runs (`/api/runs`) and of the recorded sample streams (`ui/sample_events.json`, `ui/sample_events_data_hall.json`), for a demo that doesn't depend on the network.

## 11. Outputs

All outputs go under `runs/<run_id>/`:

| Path | Content |
|---|---|
| `events.jsonl` | The full event stream (the run's ledger) |
| `references.json` | Research results, if a You.com key was set |
| `spec.json` | The agent-written Definition of Done |
| `iter_NN/layout.py` | The program the agent wrote |
| `iter_NN/sandbox/` | The sandbox working directory (bundle and `result.json`) |
| `iter_NN/scene.json`, `verify.json` | Resulting scene and full verifier report |
| `iter_NN/views.png`, `vision.json` | The drawings the inspector saw, and its verdict with adjudication |
| `final/layout.usda` | OpenUSD ASCII stage, Z-up, meters and kg. It contains a `PhysicsScene` with gravity. Fixed equipment gets `PhysicsCollisionAPI`. Movable floor equipment gets `PhysicsCollisionAPI`, `PhysicsRigidBodyAPI` and `PhysicsMassAPI`, with `physics:mass` where known (racks). Zones and overhead items are visual only. Custom `factory:*` attributes record kind, fixed, model, kW, CDU, busway and reach. It loads in usdview, Omniverse or Isaac Sim |
| `final/scene.json` | The final scene |
| `final/DONE.md` | The certificate: VERIFIED or ESCALATED TO HUMAN, the goal, iteration count, token and cost usage, every check with its result and violations, the explicit assumptions, and the inspector verdict with rulings |
| `crash.txt` | The traceback, if the run crashed (status `failed`) |

## 12. Sandboxing

The agent writes code, so the code runs in isolation. Each run copies `scene.py`, `sandbox_runner.py`, `layout.py` and `base_scene.json` into a fresh working directory.

| Mode | Isolation | Notes |
|---|---|---|
| `subprocess` (default for `run.py`) | `python -I`, a whitelist of about 25 builtins, and `__import__` limited to `math`. 20 s timeout | Fast, with no dependencies. **Not a security boundary on its own** (Python-level restrictions can be escaped). It stops accidental damage, not attacks |
| `docker` | `--network none --read-only --memory 256m --cpus 0.5 --pids-limit 64 --cap-drop ALL --security-opt no-new-privileges`, `python:3.12-slim`, only the working directory mounted | Throwaway container per iteration |
| `openshell` (NVIDIA OpenShell) | One **warm** sandbox `donetwin-sbx` (`python:3.12-slim`, 1 CPU, 512 Mi) created once with `openshell/layout-sandbox-policy.yaml`: no network policies (zero egress), read-only `/usr /lib /lib64 /bin /etc`, read-write `/tmp` only, runs as user `sandbox` | Each iteration sends the 4 files as a JSON **stdin bundle** to a base64 bootstrap that unpacks them into a temp dir under `/tmp` and runs the runner. The result comes back on stdout after the `@@DONETWIN_RESULT@@` marker. On Windows it runs through WSL (`OPENSHELL_WSL_DISTRO`, default `Ubuntu`); `OPENSHELL_CMD` overrides the command |

**OpenShell status:** integrated in code, but **not yet installed on the demo machine**, so it hasn't run end to end. All saved runs so far used `subprocess`.

## 13. Status

| Status | Item |
|---|---|
| **Built and tested** (offline, scripted `MockLLM`) | The full loop on `sorting_cell`: spec normalization, layout, subprocess sandbox, verifier, rendering, adjudication path, supervisor, USD export, DONE.md. 4 saved runs each verified 10/10 in 3 iterations. Server endpoints, event persistence and replay. UI playback of both sample streams |
| **Built, not yet tested with live models** | Spec, Layout and Vision prompts on live Nemotron via Crusoe (**waiting on the API key**). You.com research (needs `YDC_API_KEY`). `data_hall` end to end: `MockLLM` only scripts the sorting cell, and the data-hall UI stream is a recorded sample, not a live run. Docker mode. OpenShell mode (pending install) |
| **In progress** | UI upgrade: data-hall visuals and graphics quality |
| **Planned** | **Benchmark:** a one-shot agent vs DoneTwin on generated hall variants (pass rate, iterations, cost), plus a **model shoot-out** on Crusoe. **Parallel best-of-N** layout candidates per iteration. A **model escalation ladder** (Lightning, then Super, then Ultra) when progress stalls. **Prompt caching** of the long shared prefix (scene, catalogs, check library). **Inference telemetry** (latency and tokens per agent, shown in the UI) |

## 14. How to run

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows; use .venv/bin/pip on Linux/macOS
cp .env.example .env                              # then fill in the keys below (never commit .env)
```

| Variable | Required | Purpose |
|---|---|---|
| `CRUSOE_API_KEY` | For live runs | Crusoe Managed Inference |
| `YDC_API_KEY` | Optional | You.com research step (skipped if empty) |
| `PLANNER_MODEL`, `VISION_MODEL`, `CRUSOE_BASE_URL` | Optional | Model and endpoint overrides |
| `SANDBOX_MODE` | Optional | `subprocess`, `docker` or `openshell` (server auto-detects otherwise) |
| `SANDBOX_IMAGE`, `OPENSHELL_*`, `PORT` | Optional | Sandbox image, OpenShell settings, server port |

```bash
python server.py                                   # 3D twin at http://localhost:8000
python run.py                                      # same loop in the terminal (default: sorting_cell)
python run.py --scenario data_hall --sandbox docker --vision every --max-iters 6
python run.py --mock                               # offline pipeline test, scripted LLM, no API calls
```

Python dependencies: `openai, matplotlib, shapely, numpy, rich, python-dotenv`.

## 15. Repo layout (implementation)

```
server.py                 HTTP server: UI, run API, event polling, replay
run.py                    terminal runner (rich console output)
requirements.txt  .env.example
donetwin/
  scene.py                scene model, catalogs, Scene API, layout_signature (stdlib only; copied into the sandbox)
  checks.py               check library, MANDATORY invariants, verify(), feeder_loads(), gap_between()
  agents.py               Spec / Layout / Vision prompts and calls, normalize_spec()
  loop.py                 Run (events + artifacts), the supervisor loop, adjudication, DONE.md
  llm.py                  Crusoe client, retries, usage and cost, JSON and code extraction
  sandbox.py              subprocess / Docker / OpenShell execution
  sandbox_runner.py       runs inside the sandbox: restricted exec of layout.py
  render.py               plan + elevation PNG for the inspector
  usd.py                  OpenUSD (.usda) export with UsdPhysics schemas
  research.py             You.com Search references
  mock.py                 scripted LLM for offline tests
scenarios/  data_hall.json  sorting_cell.json
openshell/  layout-sandbox-policy.yaml
ui/         index.html  sample_events.json  sample_events_data_hall.json
runs/       generated per run (not committed)
```

Where this goes in this repo will be decided in the follow-up PR.

## 16. Demo plan

**Live demo (3 minutes):**

| Time | Beat |
|---|---|
| 0:00 to 0:20 | The problem: "Ask an agent to fit out a data hall. It writes code, says done, and the busway is overloaded and a rack is on a column." |
| 0:20 to 0:40 | Type "Deploy 2 MW of GB200 NVL72 racks in Hall B with N+1 cooling." Click **Hand off to agents**. Step back: Human interventions 0 |
| 0:40 to 1:05 | Research pulls NEC, NFPA and GB200 sources. The Spec agent's Definition of Done appears: 14 checks, explicit assumptions, cited sources |
| 1:05 to 2:00 | Iterations in the twin. In the recorded run, iteration 1 fails capacity and hose length (the busway derate and the long south hose runs), shown in red with exact fixes. A later attempt hits a column. Supervisor decisions are shown, then all checks pass and the inspector signs off (or is overruled with measurements) |
| 2:00 to 2:35 | Thermal and X-ray views. Open `DONE.md` and `layout.usda` (physics schemas, ready for Omniverse or Isaac Sim) |
| 2:35 to 3:00 | The punchline: a one-shot agent would have stopped at iteration 1 and called it done. Recap the architecture: Crusoe, Nemotron Super and Omni, an independent verifier |

Fallback: if live inference is slow or down, use **Replay** with the recorded data-hall stream. Say on stage that it is a replay.

**1-minute video beats:** (1) 0 to 8 s: the "says done, isn't done" hook. (2) 8 to 20 s: goal in, Definition of Done out. (3) 20 to 40 s: the loop catching failures, with a re-plan or adjudication on screen. (4) 40 to 52 s: VERIFIED, the DONE.md certificate and the USD in a viewer. (5) 52 to 60 s: the tagline "Done means measured", plus the Crusoe and Nemotron logos.

## 17. Open questions and risks

- **Live model behavior is unproven.** JSON compliance, code quality and latency of Nemotron Super across 6000-token calls, and whether reasoning traces use up `max_tokens`, all need testing. The first live run is the top priority as soon as the key arrives.
- **The data hall has never run end to end.** There is no mock script for it. The 16-rack layout pushes the 60-line program limit, so the Layout agent needs to use loops.
- **Spec gaps.** If the Spec agent leaves out a `zone_clear` for a keep-out (zones don't collide), a layout can be "verified" while sitting on the slab vault. Proposed fix: inject `zone_clear` for every fixed walkway and keep-out as a mandatory invariant.
- **Adjudication is narrow.** It only measures pairwise footprint gaps. Claims about reach, sweep or alignment that name one object come out "not measurable", get upheld, and can cause false escalations. The inspector may also misread dense rack drawings (4.5 pt labels).
- **The subprocess sandbox is not secure.** Use Docker or OpenShell for anything beyond a local demo. OpenShell still needs to be installed.
- **A UI label is wrong.** The crew strip shows the Supervisor with a Nemotron Super badge, but the supervisor is deterministic code. Fix the label, or add an LLM narrator.
- **The geometry is simplified.** The checks are 2D footprints with a sector approximation of the arm sweep, not real kinematics or 3D clearance. That's acceptable for the claim we make ("verified against these checks"), and we shouldn't claim more.
- **The server has no auth and binds to loopback only.** It's fine for a local demo but isn't deployable as is.
- **Deadline:** IDEAS.md estimates submissions close at 5:00 PM PDT. Confirm with an organizer.

## 18. Relationship to existing ideas in this repo

- **IDEAS.md / IDEA-1 (Keystone, "Recursive Nemotron"):** Keystone improves Nemotron at block stacking by fine-tuning against a PyBullet verifier. DoneTwin is a **different project**: it improves one layout at inference time, with no fine-tuning. If the team submits DoneTwin, it's a **pivot** from Keystone, not an extension of it. They share ideas: a non-LLM verifier decides success, a Nemotron Super planner, a JSONL event ledger, and hard-coded supervisor rules that override the LLM (Keystone's rollback, Night Shift's verify-and-revert, DoneTwin's re-plan and escalate). Physics and geometry as ground truth is the common thread.
- **`dashboard/` and `hosting/` (verified benchmark leaderboard):** this is **complementary**. DoneTwin's planned benchmark (one-shot agent vs DoneTwin on generated hall variants) could be registered there. The payload would be a final scene JSON, scored by a small HTTP verifier that wraps `checks.verify`. The leaderboard's rule that only independently verified scores rank matches DoneTwin's thesis.
- **No file conflicts.** This PR only adds `docs/DONETWIN_SPEC.md`. Which project to submit is the team's call, and this doc doesn't decide it.

# DoneTwin: agents that don't stop until the factory layout is physically right

**Track: Done Means Done** · Crusoe Managed Inference · NVIDIA Nemotron

Ask a code-writing agent to *"insert a heavy-duty robot arm between conveyor line A and sorting bin B"* and it will
write the layout code, say "Done!", and stop. In reality, that layout puts the robot base through a structural
column, swings the arm into a second column, leaves the bin 10 cm off its fill sensor, and puts pedestrians inside the
robot's reach. The code ran. The job is not done.

DoneTwin is a team of Nemotron agents that **writes its own definition of done, verifies the result against
physics and geometry, catches itself looping, and only stops when the layout is provably safe**, or asks a human
when it can't get there.

## The loop

```
one-line goal
   │
   ▼
SPEC AGENT (Nemotron Super) ── writes the Definition of Done: explicit assumptions + machine-checkable checks
   │                            (reach, arm-swing vs columns, clearance, walkway envelope, sensor alignment, ...)
   ▼
┌─► LAYOUT AGENT (Nemotron Super) ── writes a Python layout program against the Scene API
│     ▼
│   SANDBOX ── runs the program in a locked-down container (no network, read-only, capped CPU/mem/pids)
│     ▼
│   GEOMETRY VERIFIER ── deterministic: measures every check, returns exact correction vectors
│     ▼
│   VISION INSPECTOR (Nemotron Omni) ── independent: reads only the plan + elevation drawings
│     ▼
│   ADJUDICATION ── inspector claims are re-measured; hallucinated defects are overruled with evidence
│     ▼
└── SUPERVISOR ── continue · re-plan (loop / drift detected) · done · escalate to a human
         ▼
   verified OpenUSD layout (.usda) + DONE.md certificate + live 3D twin
```

What makes it "done means done":

- **The agent writes the spec.** The Spec agent turns a vague goal into explicit assumptions ("heavy-duty = ≥300 kg →
  HD-500, 3.1 m reach") and checks. Non-negotiable invariants (no collisions, fixed structure untouched, equipment
  exists) are injected, so an agent can't "pass" by deleting the robot or moving a pillar.
- **Verification is independent of the worker.** Geometry is measured, not judged. The vision inspector never sees
  the check results, and its claims are cross-examined against measurement.
- **It notices when it's stuck.** Layout fingerprints detect loops, the penalty trend detects drift, and both trigger a
  strategy reset with higher exploration.
- **It knows when to ping a human, rarely.** That happens only when the checks and the inspector persistently disagree
  (the spec is probably missing a rule) or the budget runs out. It hands over the best attempt and the blocking checks.

## Run it

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (Windows paths)
cp .env.example .env        # add CRUSOE_API_KEY
python server.py            # 3D twin at http://localhost:8000
python run.py               # same loop in the terminal
python run.py --mock        # offline pipeline test with a scripted LLM (no API calls)
```

Artifacts per run land in `runs/<id>/`: `spec.json`, each iteration's `layout.py`, `scene.json`, `verify.json`,
`views.png`, `vision.json`, and `final/layout.usda` + `final/DONE.md`.

## Built at the hackathon

Everything in this repo was written on 2026-10-17 during the event: the scene model and Scene API, the geometry check
library, the sandbox, the agent prompts and supervisor loop, the USD exporter, and the Three.js digital twin.

## Models (Crusoe Managed Inference)

| Role | Model |
|---|---|
| Spec agent, Layout agent | `nvidia/Nemotron-3-Super-120B-A12B` |
| Vision inspector | `nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B` |

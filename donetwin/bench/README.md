# Hall benchmark: maximum IT load with 2N power, N+1 cooling and hot/cold aisles

An open-ended layout problem with an exactly verified score. A solver places GPU racks in a data hall
and assigns each one a cooling unit and two power feeds. A deterministic verifier
(`donetwin/score.py`) scores the layout from 0 to 100. There is no LLM in the scoring path.

## What a solver submits

Only decisions, never physical quantities:

```json
{"racks": [{"id": "R01", "model": "GB200-NVL72", "x": 3.3, "y": 6.2, "rot": 0,
            "cdu": "CDU_1", "feed_a": "BW_B1", "feed_b": "BW_B2"}]}
```

The verifier rebuilds the hall from the scenario plus the rack catalog, so a payload cannot change
rack power, capacities, dimensions or fixed objects. Any other key is rejected.

## Hard rules (any violation scores 0)

No collisions, in bounds, walkways and keepouts clear, 1.2 m aisles, liquid racks on a CDU and air
racks off it, CDU capacity, hose reach, two feeds per rack within tap reach, 2N power (either feed
alone carries the full rack), room air-cooling capacity, and hot/cold aisle orientation.

## Score (if all rules pass)

`70 × density + 20 × resilience + 10 × hose`

- **density:** deployed kW divided by an upper bound, `min(Σ busway usable / 2, Σ CDU capacity + air cooling)`.
- **resilience:** the share of single CDU failures the layout survives by re-homing racks to surviving
  CDUs within hose reach and capacity. This is checked by exact search.
- **hose:** shorter total hose runs score higher.

Packing in more kW leaves less spare cooling for failover, so the best layout is unknown.

## Run it

```bash
python -m unittest discover -s tests -t .                       # 26 tests
python bench.py --solver greedy --tier all --split heldout --n 5 # baseline on held-out halls
python bench.py --solver greedy --hall scenarios/data_hall_2n.json
python bench.py --solver hill_climb --tier small --split heldout --n 5   # needs CRUSOE_API_KEY
python verifier_server.py --hall scenarios/data_hall_2n.json --port 9000 # POST /evaluations
python bench/make_benchmark.py   # writes bench/hall_b_benchmark.json for the team dashboard
```

In the 3D twin (`python server.py`), pick `data_hall_2n` or a generated hall (`small-10000`, `medium-10000`,
`large-10000`) and click Run. The server runs `donetwin/hall_run.py`: the greedy baseline first, then Nemotron
proposals, each scored live. The run ends on the best verified layout.

Halls come from `donetwin/halls.py` (`generate(seed, tier)`). Train seeds are 0 to 9999 and held-out
seeds are 10000 to 10049. The held-out halls never enter the hill-climb library.

## Baseline (greedy, no LLM), measured 2026-10-17

| Set | Mean score | Feasible |
|---|---|---|
| Held-out, 5 halls per tier (15) | 67.86 | 15/15 |
| Hall B 2N (`scenarios/data_hall_2n.json`) | 68.06 (2262 of 2560 kW) | yes |

Most of the headroom is in resilience: the greedy baseline survives at most a third of CDU failures.

## Simplifications

This is a planning-grade model, not a thermal or electrical simulation. It does not model airflow or
temperatures, the share of a liquid rack's heat that goes to room air, UPS, generators, breakers,
phase balance, or hose pressure drop.

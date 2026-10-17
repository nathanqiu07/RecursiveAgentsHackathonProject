# Shared Brain (version 5)

## Active (9)

- `4c751c41` **lesson** [Checker, Layout] Hose run is Manhattan |x - port_x| + |y - port_y| <= hose_m; rows far from the CDU ports reach only x near a port, so give each rack the nearest CDU with spare capacity. (+0/-0, 0 uses, 2 evidence)
- `4ce36d48` **lesson** [Checker, Layout] Racks are 0.6 m wide at rot 0: in a row, centres sit exactly 0.6 m apart (abutting); skip any x where a pillar, panel or CDU footprint overlaps. (+0/-0, 0 uses, 3 evidence)
- `5289b34f` **lesson** [Checker, Layout, Planner] 2N: each busway carries the full kW of every rack it feeds, so a row holds at most floor(capacity_kw * derate / rack kW) liquid racks; count GB300 at 142 kW. (+0/-0, 0 uses, 4 evidence)
- `ef7fe550` **lesson** [Checker, Layout, Planner] Sum of rack kW per CDU must stay <= capacity_kw; spread liquid racks across CDUs. (+0/-0, 0 uses, 3 evidence)
- `64b3ac66` **pitfall** [all] cdu_capacity (+0/-0, 0 uses, 3 evidence)
- `891407ae` **pitfall** [all] hose (+0/-0, 0 uses, 2 evidence)
- `9e3d4e7b` **pitfall** [all] no_improvement (+0/-0, 0 uses, 4 evidence)
- `fbdc223c` **pitfall** [all] power_2n (+0/-0, 0 uses, 4 evidence)
- `fec7994d` **pitfall** [all] no_collision (+0/-0, 0 uses, 3 evidence)

## Disputed (0)


Retired: 0

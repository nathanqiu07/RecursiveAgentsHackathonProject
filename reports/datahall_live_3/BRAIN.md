# Shared Brain (version 5)

## Active (15)

- `1ff1d6df` **lesson** [Checker, Layout] Keep the whole rack footprint (x +/- 0.3, y +/- 0.6 at rot 0) inside the floor. (+0/-0, 0 uses, 2 evidence)
- `4c751c41` **lesson** [Checker, Layout] Hose run is Manhattan |x - port_x| + |y - port_y| <= hose_m; rows far from the CDU ports reach only x near a port, so give each rack the nearest CDU with spare capacity. (+0/-0, 0 uses, 3 evidence)
- `4ce36d48` **lesson** [Checker, Layout] Racks are 0.6 m wide at rot 0: in a row, centres sit exactly 0.6 m apart (abutting); skip any x where a pillar, panel or CDU footprint overlaps. (+0/-0, 0 uses, 2 evidence)
- `5289b34f` **lesson** [Checker, Layout, Planner] 2N: each busway carries the full kW of every rack it feeds, so a row holds at most floor(capacity_kw * derate / rack kW) liquid racks; count GB300 at 142 kW. (+0/-0, 0 uses, 4 evidence)
- `d30af32e` **lesson** [Checker, Layout] Each rack takes feed_a = BW_<row>1 and feed_b = BW_<row>2 of its own row and sits at that row's row_y, within tap_m of both busways and inside their x range. (+0/-0, 0 uses, 3 evidence)
- `ec3de630` **lesson** [Checker, Layout, Planner] Total HGX-B200 air-cooled kW must stay <= the room's air_cooling_kw. (+0/-0, 0 uses, 1 evidence)
- `ef7fe550` **lesson** [Checker, Layout, Planner] Sum of rack kW per CDU must stay <= capacity_kw; spread liquid racks across CDUs. (+0/-0, 0 uses, 4 evidence)
- `64b3ac66` **pitfall** [all] cdu_capacity (+0/-0, 0 uses, 4 evidence)
- `66097837` **pitfall** [all] air_capacity (+0/-0, 0 uses, 1 evidence)
- `891407ae` **pitfall** [all] hose (+0/-0, 0 uses, 3 evidence)
- `94b83f2e` **pitfall** [all] in_bounds (+0/-0, 0 uses, 2 evidence)
- `9e3d4e7b` **pitfall** [all] no_improvement (+0/-0, 0 uses, 4 evidence)
- `f103b033` **pitfall** [all] dual_feed (+0/-0, 0 uses, 3 evidence)
- `fbdc223c` **pitfall** [all] power_2n (+0/-0, 0 uses, 4 evidence)
- `fec7994d` **pitfall** [all] no_collision (+0/-0, 0 uses, 2 evidence)

## Disputed (0)


Retired: 0

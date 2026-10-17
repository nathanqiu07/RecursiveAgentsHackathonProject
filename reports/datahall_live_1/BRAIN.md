# Shared Brain (version 53)

## Active (24)

- `1ff1d6df` **lesson** [Checker, Layout] Keep the whole rack footprint (x +/- 0.3, y +/- 0.6 at rot 0) inside the floor. (+0/-0, 20 uses, 2 evidence)
- `4c751c41` **lesson** [Checker, Layout] Hose run is Manhattan |x - port_x| + |y - port_y| <= hose_m; rows far from the CDU ports reach only x near a port, so give each rack the nearest CDU with spare capacity. (+0/-0, 20 uses, 15 evidence)
- `4ce36d48` **lesson** [Checker, Layout] Racks are 0.6 m wide at rot 0: in a row, centres sit exactly 0.6 m apart (abutting); skip any x where a pillar, panel or CDU footprint overlaps. (+0/-0, 20 uses, 13 evidence)
- `5289b34f` **lesson** [Checker, Layout, Planner] 2N: each busway carries the full kW of every rack it feeds, so a row holds at most floor(capacity_kw * derate / rack kW) liquid racks; count GB300 at 142 kW. (+0/-0, 20 uses, 30 evidence)
- `697e6ae8` **lesson** [Layout] GB200/GB300 racks need cdu set to a CDU id; HGX-B200 air racks need cdu null. (+0/-0, 20 uses, 2 evidence)
- `6f91586a` **lesson** [Layout] Reply with exactly one JSON object in the required format, no prose. (+0/-0, 10 uses, 1 evidence)
- `9a8d4a50` **lesson** [Layout] The payload is only {"racks": [...]}; each rack has exactly id, model, x, y, rot, cdu, feed_a, feed_b, with ids unique and different from every hall object id. (+0/-0, 20 uses, 2 evidence)
- `d30af32e` **lesson** [Checker, Layout] Each rack takes feed_a = BW_<row>1 and feed_b = BW_<row>2 of its own row and sits at that row's row_y, within tap_m of both busways and inside their x range. (+0/-0, 20 uses, 5 evidence)
- `ec3de630` **lesson** [Checker, Layout, Planner] Total HGX-B200 air-cooled kW must stay <= the room's air_cooling_kw. (+0/-0, 20 uses, 2 evidence)
- `ef7fe550` **lesson** [Checker, Layout, Planner] Sum of rack kW per CDU must stay <= capacity_kw; spread liquid racks across CDUs. (+0/-0, 20 uses, 22 evidence)
- `f1ecc8d9` **lesson** [Checker, Layout] Keep footprints off walkway and keepout zones (centre +/- half size); egress corridors run along the south and west walls. (+0/-0, 20 uses, 5 evidence)
- `2d5f9025` **pitfall** [all] bad_json (+0/-0, 11 uses, 1 evidence)
- `465f0a51` **pitfall** [all] cooling_type (+0/-0, 20 uses, 2 evidence)
- `4c530f6d` **pitfall** [all] payload (+0/-0, 20 uses, 2 evidence)
- `4d2ffbd3` **pitfall** [all] truncated (+0/-0, 11 uses, 1 evidence)
- `64b3ac66` **pitfall** [all] cdu_capacity (+0/-0, 20 uses, 22 evidence)
- `66097837` **pitfall** [all] air_capacity (+0/-0, 20 uses, 2 evidence)
- `7aeafe11` **pitfall** [all] zones_clear (+0/-0, 20 uses, 5 evidence)
- `891407ae` **pitfall** [all] hose (+0/-0, 20 uses, 15 evidence)
- `94b83f2e` **pitfall** [all] in_bounds (+0/-0, 20 uses, 2 evidence)
- `9e3d4e7b` **pitfall** [all] no_improvement (+0/-0, 20 uses, 30 evidence)
- `f103b033` **pitfall** [all] dual_feed (+0/-0, 20 uses, 5 evidence)
- `fbdc223c` **pitfall** [all] power_2n (+0/-0, 20 uses, 30 evidence)
- `fec7994d` **pitfall** [all] no_collision (+0/-0, 20 uses, 13 evidence)

## Disputed (0)


Retired: 0

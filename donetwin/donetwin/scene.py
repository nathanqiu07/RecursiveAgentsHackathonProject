"""Factory cell scene model and the Scene API that layout programs call.

Standard library only on purpose: this file is copied into the sandbox that
executes agent-written layout programs.
"""
import copy
import json
import math

# Robot models the agents may place. Dimensions in meters, footprint = w x d.
ROBOT_CATALOG = {
    "HD-500": {"payload_kg": 500, "reach": 3.1, "min_reach": 0.7, "w": 1.1, "d": 1.1, "h": 2.6,
               "desc": "heavy-duty 6-axis arm (engine blocks, full totes, pallets)"},
    "MD-165": {"payload_kg": 165, "reach": 2.65, "min_reach": 0.5, "w": 0.9, "d": 0.9, "h": 2.2,
               "desc": "medium-payload arm (cartons, sheet metal)"},
    "LD-20": {"payload_kg": 20, "reach": 1.8, "min_reach": 0.3, "w": 0.6, "d": 0.6, "h": 1.6,
              "desc": "light arm (small parts, pick-and-place)"},
}

# GPU racks the agents may deploy. kw = IT load, kg = loaded weight.
RACK_CATALOG = {
    "GB200-NVL72": {"kw": 132, "kg": 1360, "w": 0.6, "d": 1.2, "h": 2.3,
                    "desc": "NVIDIA GB200 NVL72, liquid-cooled, 72 Blackwell GPUs"},
    "GB300-NVL72": {"kw": 142, "kg": 1400, "w": 0.6, "d": 1.2, "h": 2.3,
                    "desc": "NVIDIA GB300 NVL72, liquid-cooled, 72 Blackwell Ultra GPUs"},
    "HGX-B200": {"kw": 15, "kg": 900, "w": 0.6, "d": 1.2, "h": 2.2,
                 "desc": "air-cooled rack of HGX B200 servers"},
}

PHYSICAL_KINDS = {"pillar", "conveyor", "bin", "robot", "fence", "rack", "pallet", "cdu", "panel"}
ZONE_KINDS = {"walkway", "keepout"}
OVERHEAD_KINDS = {"sensor", "busway"}


class SceneError(Exception):
    pass


class Scene:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.data.setdefault("objects", [])

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def to_dict(self):
        return copy.deepcopy(self.data)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

    @property
    def floor_w(self):
        return self.data["floor"]["w"]

    @property
    def floor_d(self):
        return self.data["floor"]["d"]

    def _find(self, obj_id):
        for o in self.data["objects"]:
            if o["id"] == obj_id:
                return o
        known = ", ".join(o["id"] for o in self.data["objects"])
        raise SceneError(f"unknown object '{obj_id}'. Known ids: {known}")

    def has(self, obj_id):
        return any(o["id"] == obj_id for o in self.data["objects"])

    def get(self, obj_id):
        """Return a copy of an object's record (x, y, w, d, h, rot, kind, ...)."""
        return copy.deepcopy(self._find(obj_id))

    def ids(self, kind=None):
        return [o["id"] for o in self.data["objects"] if kind is None or o["kind"] == kind]

    def point(self, ref):
        """World (x, y) of 'obj' (its center) or 'obj.name' (a named interaction point)."""
        obj_id, _, name = ref.partition(".")
        o = self._find(obj_id)
        if not name:
            return (o["x"], o["y"])
        pts = o.get("points", {})
        if name not in pts:
            raise SceneError(f"'{obj_id}' has no point '{name}'. Points: {list(pts)}")
        dx, dy = pts[name]
        a = math.radians(o.get("rot", 0.0))
        return (o["x"] + dx * math.cos(a) - dy * math.sin(a),
                o["y"] + dx * math.sin(a) + dy * math.cos(a))

    def add_robot(self, obj_id, model, x, y, rot=0.0):
        if model not in ROBOT_CATALOG:
            raise SceneError(f"unknown robot model '{model}'. Catalog: {list(ROBOT_CATALOG)}")
        if self.has(obj_id):
            raise SceneError(f"object '{obj_id}' already exists; use move()")
        spec = ROBOT_CATALOG[model]
        self.data["objects"].append({
            "id": obj_id, "kind": "robot", "model": model,
            "x": float(x), "y": float(y), "rot": float(rot),
            "w": spec["w"], "d": spec["d"], "h": spec["h"],
            "reach": spec["reach"], "min_reach": spec["min_reach"],
            "payload_kg": spec["payload_kg"], "fixed": False,
        })
        return self.get(obj_id)

    def add_rack(self, obj_id, model, x, y, rot=0.0, cdu=None, busway=None):
        if model not in RACK_CATALOG:
            raise SceneError(f"unknown rack model '{model}'. Catalog: {list(RACK_CATALOG)}")
        if self.has(obj_id):
            raise SceneError(f"object '{obj_id}' already exists; use move()/assign()")
        spec = RACK_CATALOG[model]
        self.data["objects"].append({
            "id": obj_id, "kind": "rack", "model": model,
            "x": float(x), "y": float(y), "rot": float(rot),
            "w": spec["w"], "d": spec["d"], "h": spec["h"], "kw": spec["kw"], "kg": spec["kg"],
            "cdu": cdu, "busway": busway, "fixed": False,
        })
        return self.get(obj_id)

    def assign(self, obj_id, cdu=None, busway=None):
        """Set a rack's cooling (CDU) and/or power (busway) feed."""
        o = self._find(obj_id)
        if cdu is not None:
            o["cdu"] = cdu
        if busway is not None:
            o["busway"] = busway
        return self.get(obj_id)

    def add_object(self, obj_id, kind, x, y, w, d, h, rot=0.0, points=None):
        if self.has(obj_id):
            raise SceneError(f"object '{obj_id}' already exists; use move()")
        if kind == "robot":
            raise SceneError("use add_robot() for robots")
        self.data["objects"].append({
            "id": obj_id, "kind": kind, "x": float(x), "y": float(y), "rot": float(rot),
            "w": float(w), "d": float(d), "h": float(h), "fixed": False,
            "points": points or {},
        })
        return self.get(obj_id)

    def move(self, obj_id, x=None, y=None, rot=None):
        o = self._find(obj_id)
        if x is not None:
            o["x"] = float(x)
        if y is not None:
            o["y"] = float(y)
        if rot is not None:
            o["rot"] = float(rot)
        return self.get(obj_id)

    def translate(self, obj_id, dx=0.0, dy=0.0):
        o = self._find(obj_id)
        o["x"] = float(o["x"]) + dx
        o["y"] = float(o["y"]) + dy
        return self.get(obj_id)

    def remove(self, obj_id):
        o = self._find(obj_id)
        self.data["objects"].remove(o)


def layout_signature(scene_dict, step=0.1):
    """Coarse fingerprint of the movable objects, used to detect loops."""
    sig = []
    for o in sorted(scene_dict["objects"], key=lambda o: o["id"]):
        if not o.get("fixed"):
            sig.append((o["id"], round(o["x"] / step), round(o["y"] / step), round(o.get("rot", 0))))
    return tuple(sig)

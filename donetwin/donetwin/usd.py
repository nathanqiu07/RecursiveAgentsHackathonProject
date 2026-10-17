"""Export a verified layout as OpenUSD ASCII (.usda), loadable in Omniverse / usdview."""
import re

RGB = {"rack": (0.08, 0.09, 0.1), "cdu": (0.7, 0.74, 0.78), "busway": (0.9, 0.5, 0.13),
       "keepout": (0.9, 0.3, 0.25), "panel": (0.5, 0.55, 0.55), "pillar": (0.33, 0.35, 0.37), "conveyor": (0.18, 0.43, 0.71), "bin": (0.88, 0.48, 0.10),
       "robot": (0.83, 0.63, 0.09), "sensor": (0.56, 0.27, 0.68), "walkway": (0.96, 0.82, 0.25)}


def _name(s):
    n = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return n if n[0].isalpha() else f"_{n}"


def _physics_apis(o):
    """Floor zones are visual only; equipment collides; movable equipment is a rigid body with mass (PhysX)."""
    if o["kind"] in ("walkway", "keepout") or o.get("z", 0) > 0:
        return ""
    if o.get("fixed"):
        return '"PhysicsCollisionAPI"'
    return '"PhysicsCollisionAPI", "PhysicsRigidBodyAPI", "PhysicsMassAPI"'


def to_usda(scene_dict, doc=""):
    lines = ["#usda 1.0", "(", '    defaultPrim = "Factory"', "    metersPerUnit = 1", "    kilogramsPerUnit = 1",
             '    upAxis = "Z"', f'    doc = "{doc}"', ")", "", 'def Xform "Factory"', "{",
             '    def PhysicsScene "PhysicsScene"', "    {",
             "        vector3f physics:gravityDirection = (0, 0, -1)",
             "        float physics:gravityMagnitude = 9.81", "    }"]
    for o in scene_dict["objects"]:
        h = max(o["h"], 0.01)
        z = o.get("z", 0.0) + h / 2
        r, g, b = RGB.get(o["kind"], (0.6, 0.6, 0.6))
        lines += [
            f'    def Cube "{_name(o["id"])}" (',
            f'        prepend apiSchemas = [{_physics_apis(o)}]',
            "    )",
            "    {",
            "        double size = 1",
            f"        double3 xformOp:translate = ({o['x']:.4f}, {o['y']:.4f}, {z:.4f})",
            f"        float xformOp:rotateZ = {o.get('rot', 0.0):.3f}",
            f"        float3 xformOp:scale = ({o['w']:.4f}, {o['d']:.4f}, {h:.4f})",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]',
            f"        color3f[] primvars:displayColor = [({r}, {g}, {b})]",
            f'        custom string factory:kind = "{o["kind"]}"',
            f"        custom bool factory:fixed = {1 if o.get('fixed') else 0}",
        ]
        if "kg" in o:
            lines.append(f"        float physics:mass = {o['kg']}")
        if o["kind"] == "rack":
            lines += [f'        custom string factory:model = "{o["model"]}"',
                      f"        custom double factory:kw = {o['kw']}",
                      f'        custom string factory:cdu = "{o.get("cdu")}"',
                      f'        custom string factory:busway = "{o.get("busway")}"']
        if o["kind"] == "robot":
            lines += [f'        custom string factory:model = "{o["model"]}"',
                      f"        custom double factory:reach = {o['reach']}"]
        lines.append("    }")
    lines.append("}")
    return "\n".join(lines) + "\n"

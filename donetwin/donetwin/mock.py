"""Scripted stand-in for the LLM so the pipeline can be tested offline. Never used in the real demo."""
import json

from .llm import Usage

MOCK_SPEC = {
    "interpretation": "Place a heavy-duty arm that picks parts at the east end of conveyor A and drops them into bin B.",
    "assumptions": ["'Heavy-duty' means payload >= 300 kg, so model HD-500 (3.1 m reach).",
                    "Usable reach is 90% of max reach to keep the wrist out of singularities.",
                    "Bin B must sit under overhead sensor S within 5 cm or the fill sensor misses drops.",
                    "Walkway must be outside the robot's restricted space (no guarding planned)."],
    "new_objects": [{"id": "robot_1", "kind": "robot", "model": "HD-500", "pick": "conveyor_A.pick",
                     "place": "bin_B.drop"}],
    "movable": ["bin_B"],
    "checks": [
        {"type": "reach", "params": {"robot": "robot_1", "point": "conveyor_A.pick", "margin": 0.9},
         "why": "must reach parts at the conveyor discharge"},
        {"type": "reach", "params": {"robot": "robot_1", "point": "bin_B.drop", "margin": 0.9},
         "why": "must reach the bin drop point"},
        {"type": "swing_clear", "params": {"robot": "robot_1", "from": "conveyor_A.pick", "to": "bin_B.drop",
                                           "against_kinds": ["pillar"], "arm_width": 0.4},
         "why": "arm swing must not strike structural columns"},
        {"type": "min_clearance", "params": {"subject": "robot_1", "against_kinds": ["pillar", "conveyor", "bin"],
                                             "distance": 0.5}, "why": "0.5 m maintenance access around the base"},
        {"type": "zone_clear", "params": {"zone": "walkway_main", "include_envelopes": True},
         "why": "pedestrians must stay outside the robot's restricted space"},
        {"type": "aligned", "params": {"a": "bin_B.drop", "b": "sensor_S.target", "tolerance": 0.05},
         "why": "drops must land under the fill sensor"},
    ],
    "escalate_if": ["no base position satisfies reach, swing and walkway constraints together"],
}

MOCK_PROGRAMS = [
    """px, py = scene.point("conveyor_A.pick")
bx, by = scene.point("bin_B.drop")
scene.add_robot("robot_1", "HD-500", (px + bx) / 2, (py + by) / 2)
print("Robot at the midpoint between pick and drop.")""",
    """sx, sy = scene.point("sensor_S.target")
scene.move("bin_B", x=sx, y=sy)
scene.add_robot("robot_1", "HD-500", 8.6, 6.2)
print("Moved robot south-west of pillar P5 and centered bin under sensor.")""",
    """sx, sy = scene.point("sensor_S.target")
scene.move("bin_B", x=sx, y=sy)
scene.add_robot("robot_1", "HD-500", 9.5, 8.9)
print("Robot north-east of P5; arm swings the long way around, clear of all columns.")""",
]


class MockLLM:
    def __init__(self):
        self.usage = Usage()
        self.layout_calls = 0

    def chat(self, model, messages, **kw):
        system = messages[0]["content"]
        if system.startswith("You are the SPEC AGENT"):
            return json.dumps(MOCK_SPEC)
        if system.startswith("You are the LAYOUT AGENT"):
            prog = MOCK_PROGRAMS[min(self.layout_calls, len(MOCK_PROGRAMS) - 1)]
            self.layout_calls += 1
            return f"```python\n{prog}\n```"
        return json.dumps({"verdict": "pass", "issues": []})

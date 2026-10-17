"""Small real verification example: check a file in the pinned checkout."""

import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["VERIFIER_JOB_FILE"]).read_text())
parameters = manifest["parameters"]
checkout = Path(os.environ["VERIFIER_CHECKOUT_DIR"]).resolve()
target = (checkout / parameters["path"]).resolve()
if not target.is_relative_to(checkout):
    raise ValueError("The requested file must stay inside the checkout")
exists = target.is_file()
contains = exists and parameters["contains"] in target.read_text()
print(
    json.dumps(
        {
            "schema_version": 1,
            "result": {
                "passed": contains,
                "file_exists": exists,
                "contains_expected_text": contains,
            },
        }
    )
)

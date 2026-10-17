#!/usr/bin/env python3
"""Install read-only shared pool discovery in the intentionally ignored ResourceManager."""

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install(root):
    package = Path(root) / "resource_manager"
    main = package / "main.py"
    text = main.read_text()
    if "from .resource_status import filter_available, resource_status" not in text:
        anchor = "from .resource_catalog import ResourceCatalog"
        route = """@app.get("/resources", dependencies=[Depends(authenticate)])
def resources(request: Request):
    return {"resources": request.app.state.queue.resources(catalog(request))}"""
        if text.count(anchor) != 1 or text.count(route) != 1:
            raise ValueError(
                "ResourceManager routes changed; review discovery integration before installing"
            )
        text = text.replace(
            anchor,
            anchor + "\nfrom .resource_status import filter_available, resource_status",
        )
        text = text.replace(
            route,
            """@app.get("/resources", dependencies=[Depends(authenticate)])
def resources(request: Request):
    inventory = catalog(request)
    return {"resources": filter_available(request.app.state.queue.resources(inventory), inventory)}


@app.get("/resources/status", dependencies=[Depends(authenticate)])
def resources_status(request: Request):
    return resource_status(request.app.state.queue, catalog(request))""",
        )
    compile(text, str(main), "exec")
    source = (HERE / "rm_resource_status.py").read_text()
    compile(source, "resource_status.py", "exec")
    (package / "resource_status.py").write_text(source)
    main.write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resource-manager", default=str(HERE.parent / "ResourceManager")
    )
    install(parser.parse_args().resource_manager)
    print("Installed shared pool discovery; restart the local ResourceManager API.")

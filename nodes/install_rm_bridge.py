#!/usr/bin/env python3
"""Install or refresh the tracked VM bridge in the local ResourceManager source."""

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install(root):
    package = Path(root) / "resource_manager"
    catalog = package / "resource_catalog.py"
    worker = package / "worker.py"
    catalog_text = catalog.read_text()
    worker_text = worker.read_text()
    if 'runtime: Literal["docker", "remote_vm"]' not in catalog_text:
        anchors = [
            "from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator",
            "    docker_context: str | None = None",
            "    def check_request(self, request: JobSubmission):",
        ]
        if not all(catalog_text.count(anchor) == 1 for anchor in anchors):
            raise ValueError(
                "ResourceManager catalog has changed; review the small bridge extension before applying"
            )
        catalog_text = catalog_text.replace(
            anchors[0], anchors[0] + ", model_validator"
        )
        catalog_text = catalog_text.replace(
            anchors[1],
            anchors[1]
            + """
    runtime: Literal["docker", "remote_vm"] = "docker"
    node_manifest: str | None = None""",
        )
        catalog_text = catalog_text.replace(
            anchors[2],
            """    @model_validator(mode="after")
    def approved_runtime(self):
        if self.runtime == "remote_vm":
            if not self.node_manifest or not Path(self.node_manifest).is_absolute():
                raise ValueError("remote_vm requires an absolute private node_manifest path")
            if self.docker_context is not None or self.max_concurrent_jobs != 1:
                raise ValueError("remote_vm requires one VM at a time and no Docker context")
        elif self.node_manifest is not None:
            raise ValueError("node_manifest is only valid for remote_vm")
        return self

"""
            + anchors[2],
        )
    if "def runtime_for(resource):" not in worker_text:
        anchor = (
            "def run_once(queue, catalog, worker_id, runtime_factory=DockerRuntime):"
        )
        if (
            worker_text.count(anchor) != 1
            or worker_text.count("runtime = runtime_factory(resource)") != 1
        ):
            raise ValueError(
                "ResourceManager worker has changed; review the bridge before applying"
            )
        worker_text = worker_text.replace(
            anchor,
            """def runtime_for(resource):
    if resource.runtime == "remote_vm":
        from .remote_vm_runtime import RemoteVMRuntime

        return RemoteVMRuntime(resource)
    return DockerRuntime(resource)


def run_once(queue, catalog, worker_id, runtime_factory=None):""",
        )
        worker_text = worker_text.replace(
            "runtime = runtime_factory(resource)",
            "runtime = (runtime_factory or runtime_for)(resource)",
        )
    # Validate all source edits before writing any file.
    compile(catalog_text, str(catalog), "exec")
    compile(worker_text, str(worker), "exec")
    catalog.write_text(catalog_text)
    worker.write_text(worker_text)
    shutil.copyfile(HERE / "rm_remote_runtime.py", package / "remote_vm_runtime.py")
    status_installer = HERE.parent / "dev" / "install_resource_status.py"
    if status_installer.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "install_resource_status", status_installer
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resource-manager", default=str(HERE.parent / "ResourceManager")
    )
    args = parser.parse_args()
    install(args.resource_manager)
    print("Installed remote_vm adapter; restart the local ResourceManager worker.")

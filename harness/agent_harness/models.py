from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20000)]
Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
]


class Allocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    resource_id: Identifier
    node_id: Identifier
    cpus: float = Field(gt=0, le=256, allow_inf_nan=False)
    memory_mb: int = Field(ge=64, le=1048576)
    scratch_mb: int = Field(ge=16, le=1048576)
    timeout_seconds: int = Field(ge=30, le=86400)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["researcher", "attempt"]
    objective: Text
    benchmark_id: UUID | None = None
    allocation: Allocation
    max_steps: int = Field(default=40, ge=1, le=100, strict=True)

    @model_validator(mode="after")
    def benchmark_required(self):
        if self.role == "attempt" and self.benchmark_id is None:
            raise ValueError("An attempt agent requires benchmark_id")
        return self

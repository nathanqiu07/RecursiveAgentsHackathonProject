"""Seed splits for AHC039. Train and dev use generator seeds we pick; held-out is the 150 private Lite seeds.

train   = seeds 0-19 in 4 tasks of 5 (task 0 is exactly the 5 public Lite seeds, 0-4)
dev     = seeds 20-49 in 3 tasks of 10
heldout = the 150 private Lite seeds from data.json in 5 tasks of 30

Every task's `feedback_seeds` are the only seeds whose scores the solver ever sees. For train and dev
they are the task's own seeds. For held-out they are the public seeds: the solver picks its best attempt
on those, and the chosen program is then scored on the 30 private seeds, which never reach a prompt.
"""
from __future__ import annotations

from benchmarks.alebench import scorer
from team.types import Task

TRAIN_SEEDS = list(range(0, 20))
DEV_SEEDS = list(range(20, 50))
TRAIN_TASK_SIZE, DEV_TASK_SIZE, HELDOUT_TASK_SIZE = 5, 10, 30


def public_seeds() -> list[int]:
    return [int(s) for s in scorer.metadata()["seeds"]["public_lite"]]


def private_seeds() -> list[int]:
    return [int(s) for s in scorer.metadata()["seeds"]["private_lite"]]


def all_seeds() -> list[int]:
    return TRAIN_SEEDS + DEV_SEEDS + private_seeds()


def _chunks(seeds: list[int], size: int) -> list[list[int]]:
    return [seeds[i:i + size] for i in range(0, len(seeds), size)]


def _task(split: str, i: int, n_tasks: int, seeds: list[int], feedback: list[int]) -> Task:
    if split == "heldout":
        text = f"{scorer.TITLE}: held-out batch {i + 1} of {n_tasks}, {len(seeds)} hidden cases"
    else:
        text = f"{scorer.TITLE}: {split} cases {seeds[0]}-{seeds[-1]}"
    return Task(id=f"{scorer.PROBLEM}-{split}{i}", split=split, text=text,
                payload={"problem": scorer.PROBLEM, "seeds": list(seeds), "feedback_seeds": list(feedback)})


def load_tasks() -> tuple[list[Task], list[Task], list[Task]]:
    public, private = public_seeds(), private_seeds()
    if not set(public) <= set(TRAIN_SEEDS):
        raise ValueError(f"public seeds {public} must sit inside the train seeds")
    if set(private) & set(TRAIN_SEEDS + DEV_SEEDS):
        raise ValueError("a private seed collides with a train or dev seed")
    train_chunks = [public] + _chunks([s for s in TRAIN_SEEDS if s not in public], TRAIN_TASK_SIZE)
    dev_chunks = _chunks(DEV_SEEDS, DEV_TASK_SIZE)
    held_chunks = _chunks(private, HELDOUT_TASK_SIZE)
    train = [_task("train", i, len(train_chunks), c, c) for i, c in enumerate(train_chunks)]
    dev = [_task("dev", i, len(dev_chunks), c, c) for i, c in enumerate(dev_chunks)]
    heldout = [_task("heldout", i, len(held_chunks), c, public) for i, c in enumerate(held_chunks)]
    return train, dev, heldout

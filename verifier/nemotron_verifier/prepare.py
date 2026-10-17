"""Freeze a reproducible mini quality suite and C20 workload, without a GPU."""

import argparse
import hashlib
import json
import random
from pathlib import Path

from . import MODEL, REVISION
from .config import RECIPE
from .evaluate import numeric_answer


def select_examples(mmlu_test, gsm_test):
    """Sample stable dataset row indices, then preserve their prompts and answers."""
    subjects = {}
    for index, row in enumerate(mmlu_test):
        subjects.setdefault(row["subject"], []).append((index, row))
    if len(subjects) != 57:
        raise ValueError("MMLU must contain all 57 subjects")
    rng = random.Random(42)
    mmlu_rows = []
    for subject in sorted(subjects):
        for index, row in sorted(rng.sample(subjects[subject], 1)):
            choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
            mmlu_rows.append({"subject": subject, "test_index": index,
                "prompt": "Answer the multiple-choice question with a single letter: A, B, C, or D.\n\n"
                          + f"{row['question']}\n{choices}\nAnswer:",
                "answer": "ABCD"[row["answer"]]})
    indices = sorted(random.Random(42).sample(range(len(gsm_test)), 16))
    gsm_rows = []
    for index in indices:
        row = gsm_test[index]
        answer = numeric_answer(row["answer"])
        if answer is None:
            raise ValueError("GSM8K gold answer is not numeric")
        gsm_rows.append({"test_index": index,
            "prompt": "Solve the problem step by step. End with #### and the final numeric answer.\n\n"
                      + f"Question: {row['question']}\nAnswer:", "answer": str(answer)})
    return mmlu_rows, gsm_rows


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/mini")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise SystemExit("Use an empty output directory to preserve frozen evaluation data")
    api = HfApi()
    revisions = {name: api.dataset_info(name).sha for name in
                 ["cais/mmlu", "openai/gsm8k", "Salesforce/wikitext"]}
    mmlu = load_dataset("cais/mmlu", "all", revision=revisions["cais/mmlu"], split="test")
    gsm = load_dataset("openai/gsm8k", "main", revision=revisions["openai/gsm8k"], split="test")
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                        revision=revisions["Salesforce/wikitext"], split="test")
    mmlu_rows, gsm_rows = select_examples(mmlu, gsm)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    tokens = tokenizer("\n\n".join(wiki["text"]), add_special_tokens=False)["input_ids"]
    if len(tokens) < 4096:
        raise ValueError("Wikitext test corpus has too few tokens")
    workload_rng = random.Random(42)
    starts = workload_rng.sample(range(len(tokens) - 2048 + 1), 21)
    workload = [tokens[start:start + 2048] for start in starts]
    files = {}
    for name, data in [("mmlu.json", mmlu_rows), ("gsm8k.json", gsm_rows),
                       ("ppl_tokens.json", tokens[:4096]), ("workload_tokens.json", workload)]:
        content = json.dumps(data, ensure_ascii=False).encode()
        (root / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    manifest = {"suite": "mini", "model": MODEL, "revision": REVISION, "recipe": RECIPE,
                "dataset_revisions": revisions, "mmlu_count": 57, "gsm8k_count": 16,
                "ppl_token_count": 4096, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared fixed mini suite in {root.resolve()}; no scores have been measured.")


if __name__ == "__main__":
    main()

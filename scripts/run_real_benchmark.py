from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.datasets import load_locomo
from liquid_memory_agents.datasets.longmemeval import load_longmemeval, stratified_sample
from liquid_memory_agents.embeddings import LexicalAugmentedEncoder, SentenceTransformerEncoder
from liquid_memory_agents.evaluation.metrics import exact_match, locomo_qa_score, token_f1
from liquid_memory_agents.llm import TransformersAnswerReader
from liquid_memory_agents.llm.prompts import render_prompt
from liquid_memory_agents.utils import create_run_directory, seed_everything, write_environment
from liquid_memory_agents.utils.config import load_config
from run_synthetic import (
    build_agents,
    generate,
    generate_batch,
    observe_sequence,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/longmemeval.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--examples",
        type=int,
        help="Evaluate only the first N examples for a medium-size diagnostic",
    )
    selection.add_argument(
        "--stratified-examples",
        type=int,
        help=(
            "Evaluate a deterministic proportional sample over question types "
            "after applying --start-index"
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based dataset offset; use 50 for the held-out post-development split",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=(
            "vanilla", "window", "rag", "bounded_rag",
            "pure_liquid", "lexical_only", "lalm",
            "lalm_zero_prefix", "lalm_random_prefix", "lalm_permuted_prefix",
        ),
        help="Override the evaluated memory conditions",
    )
    parser.add_argument(
        "--lexical-top-k",
        type=int,
        help="Development override for lexical-only and LALM retrieval depth",
    )
    parser.add_argument(
        "--prefix-scale",
        type=float,
        help="Development override for LALM prefix strength with lexical evidence",
    )
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(
            f"checkpoint does not exist: {args.checkpoint}. Train first and use the "
            "actual results/<run_id>/checkpoints/best.pt path."
        )
    config = load_config(args.config)
    config["run_kind"] = "evaluation"
    lexical_config = config.setdefault("liquid", {}).setdefault(
        "lexical_memory", {}
    )
    if args.lexical_top_k is not None:
        capacity = int(lexical_config.get("capacity", 512))
        if not 1 <= args.lexical_top_k <= capacity:
            parser.error(f"--lexical-top-k must be between 1 and {capacity}")
        lexical_config["top_k"] = args.lexical_top_k
    if args.prefix_scale is not None:
        if not 0.0 <= args.prefix_scale <= 1.0:
            parser.error("--prefix-scale must be between 0 and 1")
        lexical_config["prefix_scale"] = args.prefix_scale
    if args.agents:
        config.setdefault("evaluation", {})["agents"] = args.agents
    dataset_name = config["dataset"]["name"]
    if dataset_name not in {"longmemeval", "locomo"}:
        raise SystemExit(f"Unsupported real benchmark: {dataset_name}")
    dataset_path = Path(config["dataset"]["input"])
    if not dataset_path.is_file():
        parser.error(
            f"{dataset_name} data not found: {dataset_path}. Run "
            f"`python scripts/download_data.py --dataset {dataset_name}` first."
        )
    examples = (
        load_longmemeval(dataset_path)
        if dataset_name == "longmemeval"
        else load_locomo(dataset_path)
    )
    if args.start_index < 0 or args.start_index >= len(examples):
        parser.error(f"--start-index must be between 0 and {len(examples) - 1}")
    examples = examples[args.start_index:]
    if args.examples is not None:
        if args.examples < 1:
            parser.error("--examples must be positive")
        examples = examples[: args.examples]
    elif args.stratified_examples is not None:
        if not 1 <= args.stratified_examples <= len(examples):
            parser.error(
                f"--stratified-examples must be between 1 and {len(examples)}"
            )
        examples = stratified_sample(
            examples,
            args.stratified_examples,
            int(config["run"]["seed"]),
        )
    elif args.smoke:
        examples = examples[:10]
    config["execution"] = {
        "smoke": args.smoke,
        "examples": len(examples),
        "selection": (
            "stratified_by_question_type"
            if args.stratified_examples is not None
            else "contiguous"
        ),
        "selection_seed": (
            int(config["run"]["seed"])
            if args.stratified_examples is not None
            else None
        ),
        "lexical_top_k": int(lexical_config.get("top_k", 8)),
        "prefix_scale": float(lexical_config.get("prefix_scale", 0.1)),
        "checkpoint": str(args.checkpoint.resolve()),
        "agents": config.get("evaluation", {}).get("agents", []),
        "start_index": args.start_index,
        "shared_history_reuse": dataset_name == "locomo",
    }
    seed_everything(config["run"]["seed"])
    run_dir = create_run_directory(config["run"]["output_root"], dataset_name)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    write_environment(run_dir)
    semantic_encoder = SentenceTransformerEncoder(config["embedding"]["model"])
    liquid_encoder = LexicalAugmentedEncoder(
        semantic_encoder, int(config["liquid"].get("lexical_bytes", 256))
    )
    reader = TransformersAnswerReader(
        config["answer_llm"]["model"],
        max_input_tokens=config["answer_llm"]["max_input_tokens"],
        max_new_tokens=config["answer_llm"]["max_new_tokens"],
        abstention_response=config["answer_llm"].get(
            "abstention_response", "I do not know."
        ),
    )
    agents = build_agents(semantic_encoder, liquid_encoder, reader, config, args.checkpoint)
    rows, latency_rows, memory_rows = [], [], []
    hypotheses = {agent.name: [] for agent in agents}
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as output:
        def save_record(agent, example, update_ms, generated):
            prediction, context, diagnostics, read_ms, generation_ms = generated
            retrieved = set(diagnostics.get("retrieved_session_ids", []))
            gold = set(example.answer_session_ids)
            row = {
                "example_id": example.question_id, "agent": agent.name,
                "question_type": example.question_type, "query": example.question,
                "query_timestamp": example.question_date, "reference": example.answer,
                "prediction": prediction, "correct": exact_match(prediction, example.answer),
                "token_f1_diagnostic": token_f1(prediction, example.answer),
                "exact_answer_inclusion_diagnostic": example.answer.casefold() in prediction.casefold(),
                "gold_session_recall_diagnostic": (
                    len(retrieved & gold) / len(gold) if gold else None
                ),
                "prompt": render_prompt(example.question, context, example.question_date),
                "memory_context": context, "diagnostics": diagnostics,
                "official_score": None,
            }
            if dataset_name == "locomo":
                category = int(example.question_type.removeprefix("category-"))
                row["locomo_official_qa_score"] = locomo_qa_score(
                    prediction, example.answer, category
                )
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            rows.append(row)
            hypotheses[agent.name].append(
                {"question_id": example.question_id, "hypothesis": prediction}
            )
            latency_rows.append({
                "example_id": example.question_id, "agent": agent.name,
                "update_ms": update_ms, "read_ms": read_ms,
                "generation_ms": generation_ms,
                "end_to_end_ms": update_ms + read_ms + generation_ms,
                "model_loading_policy": "one shared preloaded model",
            })
            memory_rows.append({
                "example_id": example.question_id, "agent": agent.name,
                "bytes": agent.memory_size_bytes(),
            })

        def record(agent, example, update_ms):
            save_record(
                agent,
                example,
                update_ms,
                generate(agent, example.question, example.question_date),
            )

        def record_group(agent, group, update_ms):
            qa_batch_size = int(config["evaluation"].get("qa_batch_size", 1))
            amortized_update_ms = update_ms / max(1, len(group))
            starts = range(0, len(group), qa_batch_size)
            if dataset_name == "locomo":
                starts = tqdm(
                    starts,
                    total=(len(group) + qa_batch_size - 1) // qa_batch_size,
                    desc=f"{agent.name} QA batches",
                    leave=False,
                    unit="batch",
                )
            for start in starts:
                selected = group[start:start + qa_batch_size]
                generated = generate_batch(agent, selected)
                for example, result in zip(selected, generated):
                    save_record(agent, example, amortized_update_ms, result)

        for agent in agents:
            if dataset_name == "locomo":
                groups: dict[str, list] = {}
                for example in examples:
                    conversation_id = example.question_id.rsplit("-qa-", 1)[0]
                    groups.setdefault(conversation_id, []).append(example)
                grouped_examples = list(groups.values())
                for group in tqdm(
                    grouped_examples,
                    desc=f"locomo / {agent.name} conversations",
                ):
                    agent.reset()
                    observations = [
                        (turn, session["date"], session["session_id"])
                        for session in group[0].sessions
                        for turn in session["turns"]
                    ]
                    update_start = time.perf_counter()
                    observe_sequence(agent, observations)
                    update_ms = (time.perf_counter() - update_start) * 1000
                    record_group(agent, group, update_ms)
                continue
            for example in tqdm(examples, desc=f"{dataset_name} / {agent.name}"):
                agent.reset()
                update_start = time.perf_counter()
                observations = [
                    (turn, session["date"], session["session_id"])
                    for session in example.sessions
                    for turn in session["turns"]
                ]
                observe_sequence(agent, observations)
                update_ms = (time.perf_counter() - update_start) * 1000
                record(agent, example, update_ms)
    for agent, agent_rows in hypotheses.items():
        path = run_dir / f"official_input_{agent}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in agent_rows),
            encoding="utf-8",
        )
    metrics = []
    for agent in agents:
        selected = [row for row in rows if row["agent"] == agent.name]
        metrics_row = {
            "agent": agent.name, "examples": len(selected), "official_score": "",
            "diagnostic_token_f1": sum(row["token_f1_diagnostic"] for row in selected) / len(selected),
            "diagnostic_exact_match": sum(row["correct"] for row in selected) / len(selected),
        }
        if dataset_name == "locomo":
            metrics_row["locomo_official_qa_score"] = sum(
                row["locomo_official_qa_score"] for row in selected
            ) / len(selected)
        metrics.append(metrics_row)
    write_csv(run_dir / "metrics.csv", metrics)
    per_type = []
    question_types = sorted({row["question_type"] for row in rows})
    for agent in agents:
        for question_type in question_types:
            selected = [
                row for row in rows
                if row["agent"] == agent.name and row["question_type"] == question_type
            ]
            if selected:
                per_type.append({
                    "agent": agent.name,
                    "question_type": question_type,
                    "examples": len(selected),
                    "diagnostic_token_f1": sum(
                        row["token_f1_diagnostic"] for row in selected
                    ) / len(selected),
                    "diagnostic_exact_match": sum(
                        row["correct"] for row in selected
                    ) / len(selected),
                })
    write_csv(run_dir / "per_type.csv", per_type)
    write_csv(run_dir / "latency.csv", latency_rows)
    write_csv(run_dir / "memory.csv", memory_rows)
    (run_dir / "errors.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if not row["correct"]),
        encoding="utf-8",
    )
    ids = [row.question_id for row in examples]
    source_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    checkpoint_agent = next(
        (
            agent for agent in agents
            if agent.name in {
                "lalm", "pure_liquid",
                "lalm_zero_prefix", "lalm_random_prefix", "lalm_permuted_prefix",
            }
        ),
        None,
    )
    training_hashes = (
        checkpoint_agent.checkpoint_metadata["dataset_hashes"]
        if checkpoint_agent is not None
        else {}
    )
    if source_sha256 in set(training_hashes.values()):
        raise RuntimeError("real evaluation dataset hash overlaps checkpoint training data")
    (run_dir / "dataset_manifest.json").write_text(json.dumps({
        "example_ids": ids,
        "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "source_sha256": source_sha256,
        "source": str(config["dataset"]["input"]),
        "checkpoint_training_hashes": training_hashes,
    }, indent=2), encoding="utf-8")
    if dataset_name == "longmemeval":
        print(
            f"Generated all-agent hypotheses at {run_dir}. Official scores remain "
            "empty until the upstream evaluator is run."
        )
    else:
        print(
            f"Generated text-only LoCoMo QA hypotheses and local diagnostics at "
            f"{run_dir}. Released image captions were used; no image files were used."
        )


if __name__ == "__main__":
    main()

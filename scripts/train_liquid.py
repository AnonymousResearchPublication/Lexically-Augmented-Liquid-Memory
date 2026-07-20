from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liquid_memory_agents.datasets.synthetic import generate_examples, split_examples
from liquid_memory_agents.embeddings import LexicalAugmentedEncoder, SentenceTransformerEncoder
from liquid_memory_agents.liquid import (
    AdaptiveLiquidCell,
    BoundedLexicalMemory,
    QueryConditionedReader,
    StateToPrefixAdapter,
)
from liquid_memory_agents.liquid.checkpoint import load_checkpoint, save_checkpoint
from liquid_memory_agents.liquid.trainer import LiquidTrainer, TrainingExample
from liquid_memory_agents.llm.prompts import SYSTEM_PROMPT, render_prompt
from liquid_memory_agents.utils import create_run_directory, seed_everything, write_environment
from liquid_memory_agents.utils.config import load_config


BEST_CHECKPOINT_SELECTION_RULE = "maximum validation teacher-forced exact, then minimum loss"


def best_checkpoint_key(validation_metrics: dict) -> tuple[float, float]:
    return (
        float(validation_metrics.get("teacher_forced_exact", -1.0)),
        -float(validation_metrics.get("loss", float("inf"))),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--examples", type=int)
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Save an explicitly labeled random/frozen checkpoint for the trained-vs-random ablation",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    config["run_kind"] = "training"
    training_config = config.setdefault("training", {})
    epochs = args.epochs if args.epochs is not None else int(training_config.get("epochs", 5))
    example_count = (
        args.examples if args.examples is not None
        else int(training_config.get("examples", 1000))
    )
    horizons = [int(value) for value in training_config.get("horizons", [32, 64, 128, 256])]
    if not horizons or min(horizons) < 2:
        raise ValueError("training.horizons must contain values >= 2")
    training_config.update(
        {"epochs": epochs, "examples": example_count, "horizons": horizons}
    )
    lexical_config = config["liquid"].get("lexical_memory", {})
    training_lexical_context = bool(
        training_config.get("lexical_context", lexical_config.get("enabled", True))
    )
    training_config["lexical_context"] = training_lexical_context
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    run_dir = create_run_directory(config["run"]["output_root"], "train_liquid")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    write_environment(run_dir)

    semantic_encoder = SentenceTransformerEncoder(config["embedding"]["model"])
    encoder = LexicalAugmentedEncoder(
        semantic_encoder, int(config["liquid"].get("lexical_bytes", 256))
    )
    model_name = config["answer_llm"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    language_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    ).eval()
    device = next(language_model.parameters()).device
    liquid = config["liquid"]
    cell = AdaptiveLiquidCell(
        encoder.dimension, liquid["state_dim"], liquid["slots"],
        tau_min=liquid["tau_min"], tau_max=liquid["tau_max"], dt=liquid["dt"],
        adaptive_tau=liquid.get("adaptive_tau", True),
        update=liquid.get("update", "adaptive"),
        memory_decay=liquid.get("memory_decay", 0.01),
    ).to(device)
    reader = QueryConditionedReader(encoder.dimension, liquid["state_dim"]).to(device)
    adapter = StateToPrefixAdapter(
        liquid["state_dim"], language_model.get_input_embeddings().embedding_dim,
        liquid["prefix_tokens"],
    ).to(device)
    parameters = list(cell.parameters()) + list(reader.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=float(training_config.get("learning_rate", 3e-4))
    )
    correction_enabled = config.get("training", {}).get("correction_loss", True)
    trainer = LiquidTrainer(
        cell, reader, adapter, language_model, optimizer,
        correction_weight=0.1 if correction_enabled else 0.0,
        alignment_weight=float(training_config.get("alignment_weight", 0.1)),
    )
    start_epoch = 0
    resume_payload = None
    if args.resume:
        resume_payload = load_checkpoint(
            args.resume,
            {"cell": cell, "reader": reader, "adapter": adapter},
            optimizer,
            map_location=device,
        )
        start_epoch = int(resume_payload["epoch"])
        if epochs <= start_epoch:
            raise ValueError(
                f"total epochs ({epochs}) must exceed checkpoint epoch ({start_epoch})"
            )
        print(f"Resuming from epoch {start_epoch}: {args.resume}", flush=True)

    examples = []
    base_count, remainder = divmod(example_count, len(horizons))
    for horizon_index, horizon in enumerate(horizons):
        count = base_count + int(horizon_index < remainder)
        examples.extend(
            generate_examples(
                count,
                horizon=horizon,
                seed=seed + 1000 + 100_000 * horizon_index,
            )
        )
    random.Random(seed + 999).shuffle(examples)
    splits = split_examples(examples)
    ids = {name: [row.example_id for row in rows] for name, rows in splits.items()}
    manifest_records = {
        name: [
            {
                "example_id": row.example_id,
                "turns": row.turns,
                "roles": row.roles,
                "timestamps": row.timestamps,
                "session_ids": row.session_ids,
                "query": row.query,
                "answer": row.answer,
                "question_type": row.question_type,
            }
            for row in rows
        ]
        for name, rows in splits.items()
    }
    manifest_bytes = json.dumps(manifest_records, sort_keys=True).encode()
    import hashlib
    dataset_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if resume_payload is not None:
        previous_hash = resume_payload.get("dataset_hashes", {}).get(
            "generated_training_manifest"
        )
        if previous_hash != dataset_hash:
            raise RuntimeError(
                "Resume checkpoint training manifest does not match this generated split"
            )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "splits": ids,
                "content_sha256": dataset_hash,
                "counts": {name: len(rows) for name, rows in splits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.initialize_only:
        save_checkpoint(
            run_dir / "checkpoints" / "random_initialization.pt",
            {"cell": cell, "reader": reader, "adapter": adapter},
            optimizer,
            epoch=0,
            config={**config, "trained": False, "initialization_only": True},
            validation_metrics={},
            seed=seed,
            dataset_hashes={"generated_training_manifest": dataset_hash},
        )
        (run_dir / "training_metrics.json").write_text("[]\n", encoding="utf-8")
        print(f"Initialized untrained ablation checkpoint: {run_dir}")
        return

    def prepare(row):
        formatted_turns = [
            f"role={role}; timestamp={timestamp}; content={text}"
            for text, role, timestamp in zip(
                row.turns, row.roles, row.timestamps
            )
        ]
        turns, semantic_turns = encoder.encode_with_semantic(formatted_turns)
        query, semantic_query = encoder.encode_with_semantic(row.query)
        context = ""
        if training_lexical_context:
            lexical_memory = BoundedLexicalMemory(
                semantic_encoder.dimension,
                lexical_config.get("capacity", 512),
                lexical_config.get("top_k", 8),
                lexical_config.get("max_text_bytes", 1024),
                lexical_config.get("redundancy_threshold", 0.82),
                lexical_config.get("correction_cues"),
            )
            lexical_texts = [
                f"session={session_id or 'none'}; {encoded_turn}"
                for encoded_turn, session_id in zip(formatted_turns, row.session_ids)
            ]
            lexical_memory.update_many(lexical_texts, semantic_turns)
            context, _ = lexical_memory.retrieve(semantic_query)
        turns = turns.to(device)
        query = query.to(device)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_prompt(row.query, context)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt = tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=config["answer_llm"]["max_input_tokens"],
        ).input_ids.to(device)
        answer_ids = tokenizer(
            row.answer, return_tensors="pt", add_special_tokens=False
        ).input_ids
        if tokenizer.eos_token_id is not None:
            eos = torch.tensor([[tokenizer.eos_token_id]], dtype=answer_ids.dtype)
            answer_ids = torch.cat((answer_ids, eos), dim=1)
        answer = answer_ids.to(device)
        answer_embedding = encoder.encode(row.answer).to(device)
        stale = None
        if row.question_type in {"correction", "contradictory_update", "response_style"}:
            # The first turn contains the superseded value; contrast it with the current answer.
            stale_text = row.turns[0].rsplit(" ", 1)[-1].rstrip(".")
            stale = encoder.encode(stale_text).to(device)
        return TrainingExample(turns, query, prompt, answer, answer_embedding, stale)

    step_metrics = []
    if resume_payload is not None:
        previous_validation = resume_payload.get("validation_metrics", {})
        best_key = best_checkpoint_key(previous_validation)
        # Preserve the resumed state as a candidate. Without this, a worse
        # first resumed epoch would incorrectly become the new run's best.
        save_checkpoint(
            run_dir / "checkpoints" / "best.pt",
            {"cell": cell, "reader": reader, "adapter": adapter},
            optimizer,
            epoch=start_epoch,
            config={
                **config,
                "selection_rule": BEST_CHECKPOINT_SELECTION_RULE,
                "resumed_from": str(args.resume),
            },
            validation_metrics=previous_validation,
            seed=seed,
            dataset_hashes={"generated_training_manifest": dataset_hash},
        )
    else:
        best_key = (float("-inf"), -1.0)
    for epoch in range(start_epoch, epochs):
        progress = tqdm(splits["train"], desc=f"train epoch {epoch + 1}/{epochs}")
        for row in progress:
            metrics = trainer.step(prepare(row))
            step_metrics.append(metrics)
            progress.set_postfix(loss=f"{metrics['loss']:.4f}", checkpoint=run_dir.name)
        with torch.no_grad():
            validation = [trainer.loss(prepare(row)) for row in splits["validation"]]
        validation_metrics = {
            "loss": (
                sum(float(loss) for loss, _ in validation) / len(validation)
                if validation else float("nan")
            ),
            "token_accuracy": (
                sum(metrics["token_accuracy"] for _, metrics in validation) / len(validation)
                if validation else float("nan")
            ),
            "teacher_forced_exact": (
                sum(metrics["teacher_forced_exact"] for _, metrics in validation)
                / len(validation) if validation else float("nan")
            ),
        }
        save_checkpoint(
            run_dir / "checkpoints" / f"epoch_{epoch + 1}.pt",
            {"cell": cell, "reader": reader, "adapter": adapter},
            optimizer,
            epoch=epoch + 1,
            config=config,
            validation_metrics=validation_metrics,
            seed=seed,
            dataset_hashes={"generated_training_manifest": dataset_hash},
        )
        selection_key = best_checkpoint_key(validation_metrics)
        if selection_key > best_key:
            best_key = selection_key
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                {"cell": cell, "reader": reader, "adapter": adapter},
                optimizer,
                epoch=epoch + 1,
                config={
                    **config,
                    "selection_rule": BEST_CHECKPOINT_SELECTION_RULE,
                },
                validation_metrics=validation_metrics,
                seed=seed,
                dataset_hashes={"generated_training_manifest": dataset_hash},
            )
    (run_dir / "training_metrics.json").write_text(
        json.dumps(step_metrics, indent=2), encoding="utf-8"
    )
    print(f"Training complete: {run_dir}")


if __name__ == "__main__":
    main()

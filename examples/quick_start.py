# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Programmatic quick-start.

Demonstrates running all three stages from Python (rather than the CLI). The
example uses small Qwen2.5 models so it can be exercised on a single GPU.

Run::

    python examples/quick_start.py --train-data ./data/cot_sft_120k.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_compress.distill import GroupwiseDistillTrainer
from qwen_compress.qat import QADTrainer, export_quantized_model
from qwen_compress.utils.config import (
    DataConfig,
    DistillConfig,
    OptimizerConfig,
    QATConfig,
    SchedulerConfig,
    TrainingConfig,
)
from qwen_compress.utils.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Path to CoT SFT JSONL.")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--student", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--output", default="./checkpoints/quickstart")
    args = parser.parse_args()

    configure_logging(level="INFO")

    common_opt = OptimizerConfig(lr=1e-4, weight_decay=0.01)
    common_sched = SchedulerConfig(name="cosine_with_warmup", warmup_ratio=0.03)
    common_train_cfg = TrainingConfig(
        output_dir=str(Path(args.output) / "distill"),
        num_train_epochs=1.0,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=200,
        logging_steps=10,
        save_total_limit=2,
        optimizer=common_opt,
        scheduler=common_sched,
    )
    data_cfg = DataConfig(
        train_path=args.train_data,
        max_seq_length=2048,
        batch_size=1,
        gradient_accumulation_steps=16,
        num_workers=2,
        cot_mode="dual",
    )

    distill_cfg = DistillConfig(
        teacher_model_name_or_path=args.teacher,
        student_model_name_or_path=args.student,
        num_groups=12,
        group_strategy="uniform",
        data=data_cfg,
        training=common_train_cfg,
    )
    print("=" * 60, "\n[1/2] Distillation\n", "=" * 60, sep="")
    distill_out = GroupwiseDistillTrainer(distill_cfg).train()

    qat_cfg = QATConfig(
        model_name_or_path=str(distill_out),
        teacher_model_name_or_path=args.teacher,
        weight_bits=8,
        activation_bits=8,
        calibration_path=args.train_data,
        nsamples_calib=256,
        use_qad=True,
        data=DataConfig(
            train_path=args.train_data,
            max_seq_length=2048,
            batch_size=1,
            gradient_accumulation_steps=8,
            num_workers=2,
            cot_mode="dual",
        ),
        training=TrainingConfig(
            output_dir=str(Path(args.output) / "qat"),
            num_train_epochs=0.2,
            bf16=True,
            gradient_checkpointing=True,
            save_steps=200,
            logging_steps=10,
            save_total_limit=2,
            optimizer=OptimizerConfig(lr=2e-5, weight_decay=0.0),
            scheduler=SchedulerConfig(name="cosine_with_warmup", warmup_ratio=0.05),
        ),
    )
    print("=" * 60, "\n[2/2] QAT + QAD\n", "=" * 60, sep="")
    trainer = QADTrainer(qat_cfg)
    qat_out = trainer.train()
    export_dir = Path(args.output) / "export"
    export_quantized_model(trainer.student, trainer.tokenizer, export_dir, fmt="safetensors")

    print(f"\n✓ Quick-start complete!\n  distilled : {distill_out}\n  qat       : {qat_out}\n  export    : {export_dir}")


if __name__ == "__main__":
    main()

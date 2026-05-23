# Copyright 2024 qwen-compress contributors
# Licensed under the Apache License, Version 2.0.
"""Top-level command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

from qwen_compress._version import __version__
from qwen_compress.utils.config import (
    DistillConfig,
    PipelineConfig,
    PruneConfig,
    QATConfig,
    load_config,
)
from qwen_compress.utils.dist import setup_distributed
from qwen_compress.utils.logging import configure_logging, get_logger

_logger = get_logger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="qwen-compress")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO")
@click.option("--log-file", type=click.Path(dir_okay=False), default=None)
@click.option("--json-logs/--no-json-logs", default=False)
def cli(log_level: str, log_file: Optional[str], json_logs: bool) -> None:
    """qwen-compress: production-grade Qwen compression toolkit."""
    import os
    rank = int(os.environ.get("RANK", "0"))
    configure_logging(level=log_level, log_file=log_file, rank=rank, json_logs=json_logs)
    setup_distributed()


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
def distill(config_path: str) -> None:
    """Run group-wise distillation (stage 1)."""
    from qwen_compress.distill import GroupwiseDistillTrainer
    cfg = load_config(config_path, stage="distill")
    assert isinstance(cfg, DistillConfig)
    _logger.info(f"distill: teacher={cfg.teacher_model_name_or_path}, student={cfg.student_model_name_or_path}")
    trainer = GroupwiseDistillTrainer(cfg)
    final_path = trainer.train()
    click.echo(f"\nDistillation complete. Output: {final_path}")


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
def prune(config_path: str) -> None:
    """Run SparseGPT pruning (stage 2) with optional channel-permutation pass."""
    import torch
    from qwen_compress.data.calibration_data import build_calibration_loader
    from qwen_compress.models.qwen_wrapper import load_qwen_model, load_qwen_tokenizer
    from qwen_compress.prune import SparseGPTPruner
    from qwen_compress.prune.recovery import run_recovery_finetune
    from qwen_compress.utils.checkpoint import save_compressed_model

    cfg = load_config(config_path, stage="prune")
    assert isinstance(cfg, PruneConfig)

    _logger.info(f"Loading model for pruning: {cfg.model_name_or_path}")
    tokenizer = load_qwen_tokenizer(cfg.model_name_or_path)
    model = load_qwen_model(
        cfg.model_name_or_path,
        dtype="bf16",
        device_map={"": cfg.device} if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    model.eval()

    calib_loader = build_calibration_loader(
        path=cfg.calibration_path,
        tokenizer=tokenizer,
        nsamples=cfg.nsamples,
        seq_length=cfg.max_seq_length,
        seed=cfg.seed,
        batch_size=1,
    )

    pruner = SparseGPTPruner(
        model=model,
        sparsity=cfg.sparsity,
        sparsity_type=cfg.sparsity_type,
        block_size=cfg.block_size,
        percdamp=cfg.percdamp,
        device=cfg.device,
    )
    errors = pruner.prune(calib_loader)
    mean_err = sum(errors.values()) / max(1, len(errors))
    _logger.info(f"Pruning mean recon error: {mean_err:.4e}")

    perm_stats = None
    if cfg.permutation is not None and cfg.permutation.enabled:
        from qwen_compress.prune.permutation import permute_model_for_2_4
        target = cfg.permutation.target
        n_pat, m_pat = (int(x) for x in target.split(":"))
        _logger.info(f"Channel permutation: target={target}, enforce={cfg.permutation.enforce_after}")
        perm_stats = permute_model_for_2_4(
            model,
            n=n_pat,
            m=m_pat,
            enforce=cfg.permutation.enforce_after,
            max_iters=cfg.permutation.max_iters,
            swaps_per_iter=cfg.permutation.swaps_per_iter,
            seed=cfg.permutation.seed,
        )
        _logger.info(
            f"Permutation across {perm_stats['num_layers']} layers: "
            f"{perm_stats['initial_alignment_pct']:.1f}% -> "
            f"{perm_stats['after_perm_alignment_pct']:.1f}% -> "
            f"{perm_stats['after_enforce_alignment_pct']:.1f}%"
        )

    output_dir = Path(cfg.output_dir)
    save_compressed_model(
        model, tokenizer, output_dir / "pruned",
        extra_meta={
            "stage": "prune",
            "sparsity": cfg.sparsity,
            "sparsity_type": cfg.sparsity_type,
            "mean_recon_error": mean_err,
            "permutation": perm_stats,
        },
    )

    if cfg.recovery_finetune:
        if cfg.recovery is None:
            raise click.UsageError("recovery_finetune requires `recovery:` block.")
        from qwen_compress.utils.config import DataConfig
        recovery_data = DataConfig(
            train_path=cfg.calibration_path,
            max_seq_length=cfg.max_seq_length,
            batch_size=1,
            gradient_accumulation_steps=8,
            num_workers=2,
            cot_mode="dual",
        )
        final = run_recovery_finetune(
            model=model,
            tokenizer_path_or_obj=tokenizer,
            data_cfg=recovery_data,
            training_cfg=cfg.recovery,
            output_dir=output_dir / "recovered",
            extra_meta={"stage": "prune+recovery", "sparsity": cfg.sparsity},
        )
        click.echo(f"\nPruning + recovery complete. Output: {final}")
    else:
        click.echo(f"\nPruning complete. Output: {output_dir / 'pruned'}")


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--export-only", is_flag=True)
def qat(config_path: str, export_only: bool) -> None:
    """Run QAT (stage 3)."""
    from qwen_compress.qat import QADTrainer, export_quantized_model
    cfg = load_config(config_path, stage="qat")
    assert isinstance(cfg, QATConfig)
    trainer = QADTrainer(cfg)
    if not export_only:
        final = trainer.train()
    else:
        final = Path(cfg.training.output_dir) / "calibrated"
        from qwen_compress.utils.checkpoint import save_compressed_model
        save_compressed_model(trainer.student, trainer.tokenizer, final)
    export_dir = Path(cfg.training.output_dir) / "export"
    export_quantized_model(
        trainer.student, trainer.tokenizer, export_dir,
        fmt="onnx" if cfg.export_format == "onnx" else "safetensors",
    )
    click.echo(f"\nQAT complete. Checkpoint: {final}\nExport: {export_dir}")


@cli.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True, dir_okay=False))
def pipeline(config_path: str) -> None:
    """Run the full distill -> prune -> qat pipeline."""
    cfg = load_config(config_path, stage="pipeline")
    assert isinstance(cfg, PipelineConfig)
    click.echo(f"Running pipeline: {cfg.name}")
    if cfg.distill is None or cfg.prune is None or cfg.qat is None:
        raise click.UsageError("pipeline requires distill/prune/qat blocks.")
    from qwen_compress.distill import GroupwiseDistillTrainer
    distill_out = GroupwiseDistillTrainer(cfg.distill).train()
    cfg.prune.model_name_or_path = str(distill_out)
    ctx = click.get_current_context()
    import tempfile
    import yaml
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fp:
        yaml.safe_dump({"stage": "prune", **cfg.prune.model_dump()}, fp)
        prune_cfg_path = fp.name
    ctx.invoke(prune, config_path=prune_cfg_path)
    cfg.qat.model_name_or_path = str(Path(cfg.prune.output_dir) / "recovered" / "final")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fp:
        yaml.safe_dump({"stage": "qat", **cfg.qat.model_dump()}, fp)
        qat_cfg_path = fp.name
    ctx.invoke(qat, config_path=qat_cfg_path, export_only=False)
    click.echo("\nFull pipeline complete.")


def main() -> None:
    try:
        cli(standalone_mode=True)
    except KeyboardInterrupt:
        click.echo("\nInterrupted.", err=True)
        sys.exit(130)


if __name__ == "__main__":
    main()

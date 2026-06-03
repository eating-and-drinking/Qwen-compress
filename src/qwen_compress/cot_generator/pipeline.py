"""
Pipeline Controller
End-to-end workflow: Load data → Self-Instruct expansion → Batch generation → Cross validation → Write
"""

import logging
import time
from pathlib import Path

import jsonlines
from tqdm import tqdm

from .generator import CoTGenerator, GeneratorConfig
from .data_loader import load_all_sources, iter_batches
from .expander import SelfInstructExpander
from .validator import CrossValidator
from .formatter import DatasetWriter, merge_cot_and_direct

logger = logging.getLogger(__name__)


def run_pipeline(pipeline_cfg: dict):
    """
    See configs/pipeline.json for pipeline_cfg structure
    """
    # ---------------------------------------------------------------
    # 0. Initialization
    # ---------------------------------------------------------------
    output_dir = pipeline_cfg.get("output_dir", "outputs/cot_dataset")
    target_total = pipeline_cfg.get("target_total", 120000)
    enable_cross_validation = pipeline_cfg.get("enable_cross_validation", False)
    flush_every = pipeline_cfg.get("flush_every", 500)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Load data sources
    # ---------------------------------------------------------------
    logger.info("=== Step 1: Loading data sources ===")
    questions = load_all_sources(pipeline_cfg["data_sources"])
    logger.info(f"Original questions: {len(questions)}")

    # ---------------------------------------------------------------
    # 2. Initialize generator
    # ---------------------------------------------------------------
    logger.info("=== Step 2: Initializing model ===")
    gen_cfg = GeneratorConfig(**pipeline_cfg.get("generator", {}))
    generator = CoTGenerator(gen_cfg)

    # ---------------------------------------------------------------
    # 3. Self-Instruct expansion (if needed)
    # ---------------------------------------------------------------
    si_target = pipeline_cfg.get("self_instruct_count", 0)
    if si_target > 0:
        logger.info(f"=== Step 3: Self-Instruct expansion {si_target} items ===")
        expander = SelfInstructExpander(generator.llm, generator.tokenizer)
        new_qs = expander.expand(questions, target_count=si_target)
        questions.extend(new_qs)
        logger.info(f"Total questions after expansion: {len(questions)}")

    # ---------------------------------------------------------------
    # 4. Batch CoT generation
    # ---------------------------------------------------------------
    logger.info("=== Step 4: Batch CoT generation ===")
    writer = DatasetWriter(output_dir)
    validator = CrossValidator() if enable_cross_validation else None

    generated_count = 0
    start_time = time.time()
    pending_flush: list[dict] = []

    for batch in tqdm(
        iter_batches(questions, gen_cfg.batch_size),
        total=len(questions) // gen_cfg.batch_size + 1,
        desc="Generating CoT",
    ):
        results = generator.generate_batch(batch)

        # Cross validation (optional)
        if validator and results:
            results = validator.validate_batch(results, drop_low_confidence=False)

        for entry in results:
            writer.add(entry)
            pending_flush.append(entry)
            generated_count += 1

        # Periodic flush
        if len(pending_flush) >= flush_every:
            writer.flush()
            pending_flush.clear()
            elapsed = time.time() - start_time
            speed = generated_count / elapsed * 60
            logger.info(
                f"Progress: {generated_count}/{target_total} "
                f"| Speed: {speed:.0f} items/min "
                f"| Pass rate: {generator.stats()['pass_rate']}"
            )

        if generated_count >= target_total:
            logger.info(f"Reached target {target_total}, stopping generation")
            break

    # ---------------------------------------------------------------
    # 5. Write & Merge
    # ---------------------------------------------------------------
    logger.info("=== Step 5: Final write ===")
    stats = writer.finalize()
    mixed_path = merge_cot_and_direct(
        output_dir,
        cot_ratio=pipeline_cfg.get("cot_mix_ratio", 0.7),
    )

    total_time = time.time() - start_time
    logger.info(
        f"\n{'='*50}\n"
        f"Generation complete!\n"
        f"  Total generated: {generated_count}\n"
        f"  Dataset distribution: {stats}\n"
        f"  Mixed training set: {mixed_path}\n"
        f"  Time elapsed: {total_time/3600:.1f} hours\n"
        f"  Generator stats: {generator.stats()}\n"
        f"{'='*50}"
    )
    return stats

"""
Pipeline 主控
端到端流程：加载数据 → Self-Instruct 扩充 → 批量生成 → 交叉验证 → 写入
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
    pipeline_cfg 结构见 configs/pipeline.json
    """
    # ---------------------------------------------------------------
    # 0. 初始化
    # ---------------------------------------------------------------
    output_dir = pipeline_cfg.get("output_dir", "outputs/cot_dataset")
    target_total = pipeline_cfg.get("target_total", 120000)
    enable_cross_validation = pipeline_cfg.get("enable_cross_validation", False)
    flush_every = pipeline_cfg.get("flush_every", 500)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. 加载数据源
    # ---------------------------------------------------------------
    logger.info("=== Step 1: 加载数据源 ===")
    questions = load_all_sources(pipeline_cfg["data_sources"])
    logger.info(f"原始问题数: {len(questions)}")

    # ---------------------------------------------------------------
    # 2. 初始化生成器
    # ---------------------------------------------------------------
    logger.info("=== Step 2: 初始化模型 ===")
    gen_cfg = GeneratorConfig(**pipeline_cfg.get("generator", {}))
    generator = CoTGenerator(gen_cfg)

    # ---------------------------------------------------------------
    # 3. Self-Instruct 扩充（如需要）
    # ---------------------------------------------------------------
    si_target = pipeline_cfg.get("self_instruct_count", 0)
    if si_target > 0:
        logger.info(f"=== Step 3: Self-Instruct 扩充 {si_target} 条 ===")
        expander = SelfInstructExpander(generator.llm, generator.tokenizer)
        new_qs = expander.expand(questions, target_count=si_target)
        questions.extend(new_qs)
        logger.info(f"扩充后总问题数: {len(questions)}")

    # ---------------------------------------------------------------
    # 4. 批量生成 CoT
    # ---------------------------------------------------------------
    logger.info("=== Step 4: 批量生成 CoT ===")
    writer = DatasetWriter(output_dir)
    validator = CrossValidator() if enable_cross_validation else None

    generated_count = 0
    start_time = time.time()
    pending_flush: list[dict] = []

    for batch in tqdm(
        iter_batches(questions, gen_cfg.batch_size),
        total=len(questions) // gen_cfg.batch_size + 1,
        desc="生成 CoT",
    ):
        results = generator.generate_batch(batch)

        # 交叉验证（可选）
        if validator and results:
            results = validator.validate_batch(results, drop_low_confidence=False)

        for entry in results:
            writer.add(entry)
            pending_flush.append(entry)
            generated_count += 1

        # 定期刷盘
        if len(pending_flush) >= flush_every:
            writer.flush()
            pending_flush.clear()
            elapsed = time.time() - start_time
            speed = generated_count / elapsed * 60
            logger.info(
                f"进度: {generated_count}/{target_total} "
                f"| 速度: {speed:.0f} 条/分 "
                f"| 过滤率: {generator.stats()['pass_rate']}"
            )

        if generated_count >= target_total:
            logger.info(f"已达目标数量 {target_total}，停止生成")
            break

    # ---------------------------------------------------------------
    # 5. 写入 & 混合
    # ---------------------------------------------------------------
    logger.info("=== Step 5: 最终写入 ===")
    stats = writer.finalize()
    mixed_path = merge_cot_and_direct(
        output_dir,
        cot_ratio=pipeline_cfg.get("cot_mix_ratio", 0.7),
    )

    total_time = time.time() - start_time
    logger.info(
        f"\n{'='*50}\n"
        f"生成完成！\n"
        f"  总生成条数: {generated_count}\n"
        f"  数据集分布: {stats}\n"
        f"  混合训练集: {mixed_path}\n"
        f"  生成耗时: {total_time/3600:.1f} 小时\n"
        f"  生成器统计: {generator.stats()}\n"
        f"{'='*50}"
    )
    return stats

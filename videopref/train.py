"""3.4 CLI 训练模块。

示例::

    python train.py --data labels.json --cache-dir ./features_cache \\
        --output-dir ./checkpoints --epochs 100 --lr 1e-3 --seed 42

- 批量特征提取 + 持久化缓存（.pt），避免训练时重复计算。
- 训练期可选数据增强（随机水平翻转、颜色抖动、轻微旋转）；不影响缓存特征。
- tqdm 进度条 + tensorboard/wandb 日志。
- 仅训练 Attention Pooling 与 MLP 分类头，DINOv3 骨干全程冻结。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import config
from .augment import processor_image_size
from .backbone import load_backbone, trainable_parameter_count
from .dataset import build_augmented_train, build_train_val, collate_videos, load_labels
from .features import VideoPreferenceModel
from .model import default_config, save_checkpoint


# ---------------------------------------------------------------------------
# 日志器：tensorboard / wandb 二选一，均不可用则静默
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, log_dir=None, use_wandb=False, project="videopref"):
        self.writer = None
        self.wandb = None
        if use_wandb:
            try:
                import wandb

                wandb.init(project=project, reinit=True)
                self.wandb = wandb
            except Exception as e:
                print(f"[warn] wandb 初始化失败，回退: {e}", file=sys.stderr)
        elif log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(log_dir))
            except Exception as e:
                print(f"[warn] tensorboard 不可用，回退: {e}", file=sys.stderr)

    def log(self, metrics: dict, step: int):
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log({**metrics, "step": step})

    def close(self):
        if self.writer:
            self.writer.close()
        if self.wandb:
            self.wandb.finish()


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def evaluate(model, loader, device) -> tuple[float, float]:
    """返回 (loss, auc)。auc 需要两类都存在，否则返回 None。"""
    from sklearn.metrics import roc_auc_score

    model.eval()
    total_loss, count = 0.0, 0
    all_prob, all_label = [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for feats, mask, labels in loader:
            feats = feats.to(device)
            mask = mask.to(device)
            labels = labels.float().to(device)
            logits = model(feats, mask=mask, return_logits=True)
            loss = criterion(logits, labels)
            total_loss += loss.item() * feats.shape[0]
            count += feats.shape[0]
            all_prob.extend(torch.sigmoid(logits).cpu().tolist())
            all_label.extend(labels.cpu().tolist())
    avg_loss = total_loss / max(count, 1)
    auc = None
    if len(set(all_label)) > 1:
        auc = float(roc_auc_score(all_label, all_prob))
    return avg_loss, auc


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def run_training(args) -> Path:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[info] device = {device}")

    # 控制 CPU 线程数：训练主要在 GPU 上进行，torch 默认用满所有核会对极小算子
    # 做线程调度/自旋，导致 CPU 打满而 GPU 空转。限制线程数可显著降低 CPU 占用。
    if getattr(args, "threads", 0) and args.threads > 0:
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(args.threads)
        except RuntimeError:
            pass  # interop 线程只能在并行工作开始前设置
        print(f"[info] torch threads = {torch.get_num_threads()} (CPU cap)")

    # 种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 骨干（冻结）
    backbone, processor, feature_dim = load_backbone(args.backbone_dir, device=device)
    image_size = processor_image_size(processor)
    print(f"[info] backbone feature_dim = {feature_dim}, image_size = {image_size}")

    # 数据
    labels = load_labels(args.data)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = val_loader = None
    if args.augment:
        train_ds, val_ds = build_augmented_train(
            labels,
            cache_dir,
            backbone,
            processor,
            device,
            val_fraction=args.val_fraction,
            seed=args.seed,
            image_size=image_size,
            batch_size=args.batch_size,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_videos, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_videos, num_workers=0
        )
    else:
        train_ds, val_ds = build_train_val(
            labels,
            cache_dir,
            backbone,
            processor,
            device,
            val_fraction=args.val_fraction,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_videos, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_videos, num_workers=0
        )

    print(f"[info] train samples = {len(train_ds)}, val samples = {len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit("训练集为空：请检查 frames/ 下是否有清洗后的帧，以及 labels.json 路径是否匹配。")

    # 模型 + 优化器（仅可训练参数）
    model = VideoPreferenceModel(feature_dim=feature_dim).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[info] trainable params = {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.BCEWithLogitsLoss()

    logger = Logger(log_dir=args.log_dir, use_wandb=args.wandb)
    best_key = (float("-inf"), float("-inf"))
    best_path: Path | None = None
    label_mapping = config.LABEL_MAPPING
    ckpt_config = default_config(
        backbone_id=args.backbone_id,
        feature_dim=feature_dim,
        max_frames=config.DEFAULT_MAX_FRAMES,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss, count = 0.0, 0
        for feats, mask, labels in pbar:
            feats = feats.to(device)
            mask = mask.to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            logits = model(feats, mask=mask, return_logits=True)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.shape[0]
            count += feats.shape[0]
            pbar.set_postfix(loss=f"{running_loss / max(count, 1):.4f}")
        train_loss = running_loss / max(count, 1)
        scheduler.step()

        val_loss, val_auc = evaluate(model, val_loader, device)
        metrics = {"train_loss": train_loss, "val_loss": val_loss}
        if val_auc is not None:
            metrics["val_auc"] = val_auc
        logger.log(metrics, epoch)
        msg = f"[Epoch {epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        if val_auc is not None:
            msg += f" val_auc={val_auc:.4f}"
        print(msg)

        # 按 (val_auc, -val_loss) 联合择优保存最佳；auc 平局时选择 loss 更低者
        cur_key = (val_auc, -val_loss) if val_auc is not None else (float("-inf"), -val_loss)
        if cur_key > best_key:
            best_key = cur_key
            stats = {"epoch": epoch, "val_loss": val_loss}
            if val_auc is not None:
                stats["val_auc"] = val_auc
            best_path = output_dir / config.CHECKPOINT_FILENAME
            save_checkpoint(best_path, model, ckpt_config, label_mapping, stats)

    logger.close()

    # 始终再存一个最终版
    final_path = output_dir / f"final_epoch{args.epochs}.ckpt"
    stats = {"epoch": args.epochs, "train_loss": train_loss}
    if val_auc is not None:
        stats["val_auc"] = val_auc
    save_checkpoint(final_path, model, ckpt_config, label_mapping, stats)

    print(f"\n[done] best checkpoint -> {best_path}")
    print(f"[done] final checkpoint -> {final_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="个人视频喜好二分类器 — CLI 训练")
    p.add_argument("--data", required=True, help="标注文件 labels.json")
    p.add_argument("--cache-dir", default=str(config.FEATURES_CACHE_DIR), help="特征缓存目录")
    p.add_argument("--output-dir", default=str(config.CHECKPOINTS_DIR), help="Checkpoint 输出目录")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16, help="训练/特征提取 batch 大小（调大可提升 GPU 利用率）")
    p.add_argument("--threads", type=int, default=8, help="torch CPU 线程数上限（默认 8，降低 CPU 占用；0=不限制）")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--augment", action="store_true", help="训练期对帧应用数据增强")
    p.add_argument("--log-dir", default=None, help="tensorboard 日志目录")
    p.add_argument("--wandb", action="store_true", help="使用 wandb 日志")
    p.add_argument("--device", default=None, help="torch device，如 cuda:0 / cpu")
    p.add_argument("--backbone-dir", default=str(config.DEFAULT_BACKBONE_DIR), help="本地骨干权重目录")
    p.add_argument("--backbone-id", default=config.DEFAULT_BACKBONE_ID, help="骨干标识（写入 Checkpoint）")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()

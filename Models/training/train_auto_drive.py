"""
Main training loop for AutoDrive.

Responsibilities
----------------
- Build ZOD AutoDrive dataset (85 / 10 / 5 train / val / test split at sequence level)
- Run training loop with gradient-accumulation-based batch simulation
- Validate after every epoch and log to TensorBoard
- Run test evaluation at the end of training
- Save model checkpoint after every epoch

Usage
-----
python Models/training/train_auto_drive.py \
    --root /path/to/zod \
    --save-dir /path/to/checkpoints \
    [--checkpoint /path/to/resume.pth]

TensorBoard
-----------
tensorboard --logdir runs/
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[2]))
from Models.data_utils.load_data_auto_drive import LoadDataAutoDrive
from Models.training.auto_drive_trainer import AutoDriveTrainer


def collate_fn(batch):
    """Stack variable-type GT values into tensors."""
    img_prev_list, img_curr_list, d_list, c_list, f_list = zip(*batch)
    import torch
    return (
        torch.stack(img_prev_list),
        torch.stack(img_curr_list),
        torch.tensor(d_list, dtype=torch.float32),
        torch.tensor(c_list, dtype=torch.float32),
        torch.tensor(f_list, dtype=torch.float32),
    )


def run_epoch_val(trainer, loader, desc="Val"):
    """Run one full pass over loader without gradients; return averaged losses."""
    total = dist = curv = flag = 0.0
    n = 0
    for img_prev, img_curr, d_gt, c_gt, f_gt in loader:
        t, d, c, f = trainer.validate(img_prev, img_curr, d_gt, c_gt, f_gt)
        total += t; dist += d; curv += c; flag += f
        n += 1
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    return total / n, dist / n, curv / n, flag / n


def main():
    parser = ArgumentParser()
    parser.add_argument("--root",       required=True,
                        help="ZOD dataset root (contains labels/, images_blur_*/)")
    parser.add_argument("--save-dir",   default="checkpoints",
                        help="directory where .pth checkpoints are saved")
    parser.add_argument("--checkpoint", default="",
                        help="resume from existing checkpoint")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    data = LoadDataAutoDrive(args.root)

    train_loader = DataLoader(
        data.train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        data.val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        data.test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn,
    )

    n_train = len(data.train)
    n_val   = len(data.val)
    n_test  = len(data.test)
    print(f"Train: {n_train}  Val: {n_val}  Test: {n_test} samples")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = AutoDriveTrainer(checkpoint_path=args.checkpoint)
    trainer.zero_grad()

    # Gradient accumulation: simulate batch_size=16 regardless of loader batch
    accumulate_every = max(1, 16 // args.batch_size)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Learning-rate schedule
        if epoch < 2:
            trainer.set_learning_rate(5e-4)
        elif epoch < 15:
            trainer.set_learning_rate(1e-4)
        else:
            trainer.set_learning_rate(2.5e-5)

        trainer.set_train_mode()

        for batch_idx, (img_prev, img_curr, d_gt, c_gt, f_gt) in enumerate(train_loader):
            trainer.set_batch(img_prev, img_curr, d_gt, c_gt, f_gt)
            trainer.run_model()
            trainer.loss_backward()

            # Optimizer step via gradient accumulation
            if (batch_idx + 1) % accumulate_every == 0:
                trainer.run_optimizer()

            global_step += 1

            # Log scalar losses every 100 steps
            if global_step % 100 == 0:
                trainer.log_train_loss(global_step)
                print(
                    f"  step {global_step:6d}  "
                    f"loss {trainer.get_loss():.4f}  "
                    f"(d {trainer.loss_distance:.4f}  "
                    f"c {trainer.loss_curvature:.4f}  "
                    f"f {trainer.loss_flag:.4f})"
                )

            # Save a sample visualization every 500 steps
            if global_step % 500 == 0:
                trainer.save_visualization(global_step)

        # Flush any leftover gradients at epoch end
        trainer.run_optimizer()

        # ------------------------------------------------------------------
        # Save checkpoint
        # ------------------------------------------------------------------
        ckpt_path = os.path.join(
            args.save_dir,
            f"AutoDrive_epoch{epoch + 1:03d}_step{global_step}.pth"
        )
        trainer.save_model(ckpt_path)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        print("  Validating...")
        trainer.set_eval_mode()
        with torch.no_grad():
            v_total, v_dist, v_curv, v_flag = run_epoch_val(trainer, val_loader)
        trainer.log_val_loss(v_total, v_dist, v_curv, v_flag, epoch + 1)
        print(
            f"  [Val] total {v_total:.4f}  "
            f"d {v_dist:.4f}  c {v_curv:.4f}  f {v_flag:.4f}"
        )
        trainer.set_train_mode()

    # ------------------------------------------------------------------
    # Test evaluation (once, at end of training)
    # ------------------------------------------------------------------
    print("\nRunning test set evaluation...")
    trainer.set_eval_mode()
    with torch.no_grad():
        t_total, t_dist, t_curv, t_flag = run_epoch_val(trainer, test_loader, desc="Test")
    trainer.log_test_loss(t_total, t_dist, t_curv, t_flag)
    print(
        f"[Test] total {t_total:.4f}  "
        f"d {t_dist:.4f}  c {t_curv:.4f}  f {t_flag:.4f}"
    )

    trainer.cleanup()


if __name__ == "__main__":
    main()

# AutoDrive trainer
# Receives batches from the DataLoader
# Runs AutoDrive network (shared backbone → conv head → three task branches)
# Calculates per-task losses and combined loss
# Logs scalars and sample visualizations to TensorBoard

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[2]))
from Models.model_components.autodrive.autodrive_network import AutoDrive


class AutoDriveTrainer:
    """
    Wraps AutoDrive for a single training step.

    Losses
    ------
    distance  : L1( sigmoid_pred, d_norm_gt )       d_norm = (150 - min(d,150)) / 150
    curvature : L1( raw_pred, curvature_gt )
    flag      : BCEWithLogitsLoss( logits, flag_gt )  flag_gt ∈ {0, 1}
    total     : distance_loss + curvature_loss + flag_loss
    """

    def __init__(self, checkpoint_path: str = ''):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'AutoDriveTrainer using {self.device}')

        self.model = AutoDrive().to(self.device)

        if checkpoint_path:
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            )
            print(f'Loaded checkpoint: {checkpoint_path}')

        self.writer = SummaryWriter()

        self.learning_rate = 5e-4
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self._l1  = nn.L1Loss()
        self._bce = nn.BCEWithLogitsLoss()

        # State filled by set_batch / run_model
        self.loss:          torch.Tensor | None = None
        self.loss_distance: float = 0.0
        self.loss_curvature: float = 0.0
        self.loss_flag:     float = 0.0

        self._img_prev_np = None   # kept for visualizations (first sample in batch)

    # ------------------------------------------------------------------
    # Batch management
    # ------------------------------------------------------------------

    def set_batch(self, img_prev, img_curr, d_norm_gt, curvature_gt, flag_gt):
        """
        Accept tensors already from a DataLoader (B, 3, H, W) / (B,).
        Moves everything to the active device.
        """
        self.img_prev    = img_prev.to(self.device)
        self.img_curr    = img_curr.to(self.device)

        self.d_norm_gt   = d_norm_gt.float().unsqueeze(1).to(self.device)    # (B,1)
        self.curvature_gt = curvature_gt.float().unsqueeze(1).to(self.device) # (B,1)
        self.flag_gt     = flag_gt.float().unsqueeze(1).to(self.device)      # (B,1)

        # Save first sample image for TensorBoard visualization (unnorm approx)
        self._img_prev_np = img_prev[0].permute(1, 2, 0).cpu().numpy()

    def set_learning_rate(self, lr: float):
        self.learning_rate = lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    # ------------------------------------------------------------------
    # Forward + loss
    # ------------------------------------------------------------------

    def run_model(self):
        d_pred, curv_pred, flag_logits = self.model(self.img_prev, self.img_curr)

        loss_d    = self._l1(d_pred, self.d_norm_gt)
        loss_c    = self._l1(curv_pred, self.curvature_gt)
        loss_f    = self._bce(flag_logits, self.flag_gt)

        self.loss           = loss_d + loss_c + loss_f
        self.loss_distance  = loss_d.item()
        self.loss_curvature = loss_c.item()
        self.loss_flag      = loss_f.item()

        self._d_pred_sample    = d_pred[0].item()
        self._curv_pred_sample = curv_pred[0].item()
        self._flag_pred_sample = torch.sigmoid(flag_logits[0]).item()

    # ------------------------------------------------------------------
    # Validation (no_grad already ensured by caller)
    # ------------------------------------------------------------------

    def validate(self, img_prev, img_curr, d_norm_gt, curvature_gt, flag_gt):
        self.set_batch(img_prev, img_curr, d_norm_gt, curvature_gt, flag_gt)
        self.run_model()
        return self.loss.item(), self.loss_distance, self.loss_curvature, self.loss_flag

    # ------------------------------------------------------------------
    # Gradient helpers
    # ------------------------------------------------------------------

    def loss_backward(self):
        self.loss.backward()

    def run_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_loss(self) -> float:
        return self.loss.item()

    # ------------------------------------------------------------------
    # Mode helpers
    # ------------------------------------------------------------------

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str):
        print(f'Saving model → {path}')
        torch.save(self.model.state_dict(), path)

    # ------------------------------------------------------------------
    # TensorBoard logging
    # ------------------------------------------------------------------

    def log_train_loss(self, step: int):
        self.writer.add_scalar("Loss/train_total",     self.get_loss(),       step)
        self.writer.add_scalar("Loss/train_distance",  self.loss_distance,    step)
        self.writer.add_scalar("Loss/train_curvature", self.loss_curvature,   step)
        self.writer.add_scalar("Loss/train_flag",      self.loss_flag,        step)

    def log_val_loss(self, total, dist, curv, flag, epoch: int):
        self.writer.add_scalar("Loss/val_total",     total, epoch)
        self.writer.add_scalar("Loss/val_distance",  dist,  epoch)
        self.writer.add_scalar("Loss/val_curvature", curv,  epoch)
        self.writer.add_scalar("Loss/val_flag",      flag,  epoch)

    def log_test_loss(self, total, dist, curv, flag):
        self.writer.add_scalar("Loss/test_total",     total, 0)
        self.writer.add_scalar("Loss/test_distance",  dist,  0)
        self.writer.add_scalar("Loss/test_curvature", curv,  0)
        self.writer.add_scalar("Loss/test_flag",      flag,  0)

    def save_visualization(self, step: int):
        """Log current frame with pred vs GT annotations to TensorBoard."""
        d_pred_m = 150.0 * (1.0 - self._d_pred_sample)
        d_gt_m   = 150.0 * (1.0 - self.d_norm_gt[0].item())

        # Clamp display to reasonable range (unnormalized image may look washed-out)
        img_disp = self._img_prev_np.clip(0, 1)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axis('off')
        ax.imshow(img_disp)
        txt = (
            f"d_pred: {d_pred_m:.1f} m  |  d_gt: {d_gt_m:.1f} m\n"
            f"curv_pred: {self._curv_pred_sample:.4f}  |  "
            f"curv_gt: {self.curvature_gt[0].item():.4f}\n"
            f"flag_pred: {self._flag_pred_sample:.2f}  |  "
            f"flag_gt: {self.flag_gt[0].item():.0f}"
        )
        ax.text(
            10, 30, txt,
            color="black", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="black", boxstyle="round,pad=0.4"),
        )
        self.writer.add_figure("Visualization/sample", fig, global_step=step)
        plt.close(fig)

    def cleanup(self):
        self.writer.flush()
        self.writer.close()
        print('AutoDriveTrainer: finished.')

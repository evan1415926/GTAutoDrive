"""Training loop with early stopping, LR scheduling, and AMP mixed precision."""
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import numpy as np
from config.settings import TrainConfig, LABELS

torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, model, train_loader, val_loader, config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=config.lr_patience,
            factor=0.5, min_lr=1e-5
        )
        self.scaler = GradScaler('cuda')

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

    def train(self) -> dict:
        """Run full training. Returns history dict."""
        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        print(f"Training on {self.device}  (AMP: enabled)")
        print(f"Train batches: {len(self.train_loader)}, "
              f"Val batches: {len(self.val_loader)}")

        for epoch in range(self.config.epochs):
            t0 = time.perf_counter()
            train_loss = self._train_epoch()
            val_loss, val_acc = self._validate()
            elapsed = time.perf_counter() - t0

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                  f"T Loss: {train_loss:.4f} | "
                  f"V Loss: {val_loss:.4f} | "
                  f"V Acc: {val_acc:.2%} | "
                  f"LR: {lr:.1e} | "
                  f"Time: {elapsed:.1f}s")

            # Early stopping + checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone()
                                   for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1} "
                          f"(best: {self.best_epoch})")
                    break

        print(f"Best val loss: {self.best_val_loss:.4f} "
              f"at epoch {self.best_epoch}")
        history['best_state'] = self.best_state
        return history

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            with autocast('cuda'):
                loss = self.criterion(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def _validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        per_class_correct = [0] * len(LABELS)
        per_class_total = [0] * len(LABELS)
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                with autocast('cuda'):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                for i in range(len(LABELS)):
                    mask = (y == i)
                    per_class_total[i] += mask.sum().item()
                    per_class_correct[i] += (preds[mask] == i).sum().item()
        acc = correct / total
        parts = []
        for i, label in enumerate(LABELS):
            if per_class_total[i] > 0:
                pacc = per_class_correct[i] / per_class_total[i]
                parts.append(f"{label}:{pacc:.0%}")
        print(f"  Per-class: {' '.join(parts)}")
        return total_loss / len(self.val_loader), acc

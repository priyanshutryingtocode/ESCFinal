"""Reusable Level-2 specialist training logic extracted from esc-1.ipynb cell 204.

Pass notebook-equivalent specialist config, loaders, model factory, class names,
and device. This retains the warm-up/full fine-tuning procedure.
"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def evaluate_specialist(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return accuracy, macro_f1, np.array(all_targets), np.array(all_preds)


def train_specialist(key, specialist_config, loaders, class_names,
                    model_factory, device, checkpoint_dir,
                    warmup_epochs=3, max_epochs=30, patience=7,
                    backbone_lr=1e-5, classifier_lr=3e-4, weight_decay=1e-4):
    group, architecture, condition = specialist_config[key]
    train_loader, val_loader = loaders[key]["train"], loaders[key]["val"]
    num_classes = len(class_names[group])
    model = model_factory(architecture, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    for p in model.features.parameters():
        p.requires_grad = False
    optimizer = AdamW(model.classifier.parameters(), lr=classifier_lr,
                     weight_decay=weight_decay)
    scheduler = None
    best_f1, best_epoch, best_state = -1.0, 0, None
    epochs_without_improvement, history = 0, []

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"l2_{key}_source_independent.pt")

    print(f"\nTRAINING SPECIALIST: {key} | {architecture} | {condition}")
    for epoch in range(1, max_epochs + 1):
        if epoch == warmup_epochs + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            optimizer = AdamW([{"params": model.features.parameters(), "lr": backbone_lr},
                               {"params": model.classifier.parameters(), "lr": classifier_lr}],
                              weight_decay=weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
            print("Backbone unfrozen: full fine-tuning begins.")

        model.train()
        if epoch <= warmup_epochs:
            model.features.eval()  # keep BatchNorm running stats fixed during warm-up
        running_loss, train_preds, train_targets = 0.0, [], []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            train_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
            train_targets.extend(yb.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        val_acc, val_f1, _, _ = evaluate_specialist(model, val_loader, device)
        if scheduler is not None:
            scheduler.step(val_f1)
        history.append(dict(epoch=epoch, train_loss=train_loss,
                           train_accuracy=train_acc, val_accuracy=val_acc,
                           val_macro_f1=val_f1))

        improved = val_f1 > best_f1
        if improved:
            best_f1, best_epoch = val_f1, epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save({"model_state_dict": best_state, "group": group,
                        "architecture": architecture, "condition": condition,
                        "classes": class_names[group], "best_epoch": best_epoch,
                        "best_val_accuracy": val_acc, "best_val_macro_f1": best_f1,
                        "history": history}, checkpoint_path)
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch:02d}/{max_epochs} | Loss {train_loss:.4f} | "
              f"Train Acc {train_acc:.2%} | Val Acc {val_acc:.2%} | "
              f"Val Macro-F1 {val_f1:.4f}{' *' if improved else ''}")
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for specialist {key}")
    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} | Best validation Macro-F1: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    return model, history, checkpoint_path

"""Level-1 training loop from esc-1.ipynb corrected source-independent experiment.

Dataset/model construction is intentionally external: pass the notebook-equivalent
model, loaders, optimizer, criterion, groups, and class weights into train_l1().
"""
from pathlib import Path
import copy
import torch
from sklearn.metrics import accuracy_score, f1_score


def train_l1(model, train_loader, val_loader, optimizer, criterion, device,
             groups, class_weights, checkpoint_path,
             max_epochs=40, patience=10):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_macro_f1 = -1.0
    best_val_accuracy = 0.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        model.eval()
        val_loss_sum, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_preds.extend(outputs.argmax(1).cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        val_loss = val_loss_sum / len(val_targets)
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds,
                                labels=list(range(len(groups))),
                                average="macro", zero_division=0)
        history.append(dict(epoch=epoch+1, train_loss=train_loss,
                           train_accuracy=train_accuracy, val_loss=val_loss,
                           val_accuracy=val_accuracy,
                           val_macro_f1=val_macro_f1))

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | "
              f"Train: {train_accuracy:.2%} | Val Loss: {val_loss:.4f} | "
              f"Val: {val_accuracy:.2%} | Macro-F1: {val_macro_f1:.4f}")

        # Notebook cell 189 selects the best checkpoint by validation Macro-F1.
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    if best_state is None:
        raise RuntimeError("No valid Level-1 checkpoint was selected.")
    model.load_state_dict(best_state)

    torch.save({
        "stage": "level1", "model_name": "VGG16",
        "initialization": "ImageNet pretrained",
        "augmentation": "SpecAugment",
        "split_protocol": "ESC50 source-independent folds",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_val_macro_f1": best_val_macro_f1,
        "model_state_dict": best_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "class_weights": class_weights.detach().cpu(),
        "groups": groups, "history": history,
    }, checkpoint_path)

    print("\nCORRECTED LEVEL-1 TRAINING COMPLETE")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Best validation Macro-F1: {best_val_macro_f1:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    return model, history, str(checkpoint_path)

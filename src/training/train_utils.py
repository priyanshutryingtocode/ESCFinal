from pathlib import Path
import torch
import torch.nn as nn


def load_state_dict(checkpoint_path, device="cpu"):
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], checkpoint
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"], checkpoint

    raise RuntimeError(
        f"Unsupported checkpoint format: {checkpoint_path}"
    )


def save_checkpoint(model, path, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            **metadata
        },
        path
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1)
    }

import sys
import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.architectures import EnvironmentalCNN
from src.preprocessing.audio_preprocessing import preprocess_audio
from src.training.train_utils import load_state_dict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = PROJECT_ROOT / "models"

GROUPS = [
    "Animal",
    "Birds",
    "Natural Soundscapes",
    "Human",
    "Machine Sounds",
    "Domestic sounds",
    "Outdoor noises"
]

SPECIALISTS = {
    "Animal": ("efficientnetb2", "No Filter", "animal"),
    "Birds": ("efficientnetb0", "No Filter", "birds"),
    "Domestic sounds": ("efficientnetb2", "PCEN", "domestic_sounds"),
    "Human": ("vgg19", "Audio Crop", "human"),
    "Machine Sounds": ("vgg16", "Noise Removal", "machine_sounds"),
    "Natural Soundscapes": ("vgg16", "Noise Removal", "natural_soundscapes"),
    "Outdoor noises": ("efficientnetb2", "High Pass", "outdoor_noises")
}

ALPHA = 0.2


def load_model(checkpoint_path, model_name, num_classes):
    state_dict, checkpoint = load_state_dict(
        checkpoint_path,
        device="cpu"
    )

    model = EnvironmentalCNN(
        model_name=model_name,
        num_classes=num_classes
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()

    return model, checkpoint


def predict_logits(model, x):
    with torch.inference_mode():
        return model(x)


def run_inference(audio_path):
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    print("=" * 72)
    print("ESC-50 HIERARCHICAL SOUND CLASSIFICATION")
    print("=" * 72)
    print("Audio file:", audio_path)
    print("Device:", DEVICE)
    print("Decision rule: route to the 2 groups with highest Level-1 probability")
    print(f"Fusion rule: class score = specialist probability * group probability^{ALPHA}")
    print("If a class appears in both specialists, keep its higher fused score.")

    # ------------------------------------------------
    # LEVEL 1
    # ------------------------------------------------

    print("LEVEL 1 — GROUP CLASSIFICATION")

    l1_path = MODEL_DIR / "level1_vgg16_no_filter.pt"

    l1_model, l1_ckpt = load_model(
        l1_path,
        "vgg16",
        len(GROUPS)
    )

    # Use checkpoint class order if saved; otherwise training order.
    l1_classes = l1_ckpt.get("classes", GROUPS)

    x_l1 = preprocess_audio(
        audio_path,
        "No Filter",
        DEVICE
    )

    l1_logits = predict_logits(l1_model, x_l1)
    l1_probs = torch.softmax(l1_logits, dim=1)[0].cpu().numpy()

    if set(l1_classes) != set(GROUPS):
        raise RuntimeError(
            "Level-1 checkpoint classes do not match configured groups."
        )

    # Align checkpoint probability columns to canonical group order.
    aligned_probs = np.zeros(len(GROUPS), dtype=np.float64)

    for i, group in enumerate(l1_classes):
        aligned_probs[GROUPS.index(group)] = l1_probs[i]

    top2 = np.argsort(aligned_probs)[-2:][::-1]

    print("\nGroup probabilities (ranked):")
    for idx in np.argsort(aligned_probs)[::-1]:
        selected = "  [selected]" if idx in top2 else ""
        print(f"  {GROUPS[idx]:25s} {aligned_probs[idx] * 100:7.2f}%{selected}")

    print("\nTop-2 specialist routes:")
    for idx in top2:
        print(f"  {GROUPS[idx]} (group probability {aligned_probs[idx] * 100:.2f}%)")

    # ------------------------------------------------
    # LEVEL 2 — RUN BOTH SELECTED SPECIALISTS
    # ------------------------------------------------

    print("LEVEL 2 — SPECIALIST CLASSIFICATION")

    final_scores = {}
    specialist_outputs = {}

    for group_idx in top2:
        group = GROUPS[group_idx]
        model_name, condition, key = SPECIALISTS[group]

        checkpoint_path = (
            MODEL_DIR / "level2" /
            f"l2_{key}_source_independent.pt"
        )

        # Checkpoint metadata is the authoritative class ordering.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        classes = checkpoint.get("classes")

        if not classes:
            raise RuntimeError(
                f"Checkpoint for {group} does not contain its class list."
            )

        state_dict = checkpoint.get("model_state_dict")
        if state_dict is None:
            raise RuntimeError(
                f"Checkpoint for {group} has no model_state_dict."
            )

        model = EnvironmentalCNN(
            model_name=model_name,
            num_classes=len(classes)
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE).eval()

        x = preprocess_audio(
            audio_path,
            condition,
            DEVICE
        )

        logits = predict_logits(model, x)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

        group_probability = max(
            float(aligned_probs[group_idx]),
            1e-12
        )

        fused = probabilities * (group_probability ** ALPHA)

        print(f"\n{group} specialist (group probability {group_probability * 100:.2f}%):")
        print("  Class probabilities and fused scores:")

        order = np.argsort(probabilities)[::-1]
        for j in order[:3]:
            print(
                f"  {classes[j]:22s} "
                f"specialist={probabilities[j] * 100:6.2f}%  "
                f"fused={fused[j]:.6f}"
            )

        for cls, score in zip(classes, fused):
            final_scores[cls] = max(
                final_scores.get(cls, 0.0),
                float(score)
            )

        specialist_outputs[group] = {
            "classes": classes,
            "probabilities": probabilities.tolist()
        }

        del model

    # ------------------------------------------------
    # FINAL 50-CLASS RESULT
    # ------------------------------------------------

    if not final_scores:
        raise RuntimeError("No specialist produced a prediction.")

    predicted_class = max(
        final_scores,
        key=final_scores.get
    )

    final_order = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)

    total_score = sum(final_scores.values())
    confidence = (
        final_scores[predicted_class] / total_score
        if total_score > 0 else 0.0
    )

    print("=" * 72)
    print("FINAL PREDICTION")
    print("=" * 72)
    print("Final class scores (fused; highest score wins):")
    for cls, score in final_order[:5]:
        marker = "  <-- selected" if cls == predicted_class else ""
        print(f"  {cls:22s} {score:.6f}{marker}")
    print("Predicted class:", predicted_class)
    print("Predicted group:", next(
        group for group, (_, _, _) in SPECIALISTS.items()
        if predicted_class in specialist_outputs.get(group, {}).get("classes", [])
    ))
    print("Normalized fused share:", f"{confidence * 100:.2f}%")
    print("=" * 72)

    return {
        "audio": str(audio_path),
        "class": predicted_class,
        "normalized_fused_score": confidence,
        "top2_groups": [GROUPS[i] for i in top2],
        "group_probabilities": {
            group: float(aligned_probs[i])
            for i, group in enumerate(GROUPS)
        }
    }


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python run_inference.py <audio_file.wav>")
        print()
        print("Example:")
        print("  python run_inference.py data/test/1-19872-A-36.wav")
        sys.exit(1)

    run_inference(sys.argv[1])


if __name__ == "__main__":
    main()

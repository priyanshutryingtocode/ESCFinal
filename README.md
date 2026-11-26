# Environmental Sound Classification — ESC-50

Two-level hierarchical environmental sound classification.

## Final pipeline

- Dataset: ESC-50
- 50 sound classes
- Seven Level-1 groups
- Level-1: VGG16, No Filter
- Level-2: seven specialist CNNs
- Routing: Top-2
- Probability fusion alpha: 0.2

## Repository structure

```text
Environmental_Sound_Classification/
├── data/
│   ├── train/       # 8 example WAV files
│   ├── validation/  # 8 example WAV files
│   ├── test/        # 8 example WAV files
│   └── *_manifest.csv
├── models/
│   ├── level1_vgg16_no_filter.pt
│   └── level2/
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── training/
│   └── testing/
├── configs/
├── results/
├── run_inference.py
├── run_training.py
└── run_paths.py
```

The WAV files are representative examples only, not the complete dataset.
The CSV manifests describe the full source-independent experiment splits.

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run inference

```bash
python run_inference.py data/test/5-218494-A-22.wav
```

The script:
1. Predicts seven Level-1 group probabilities.
2. Selects the two highest-probability groups.
3. Runs the corresponding two specialists.
4. Fuses their class probabilities with alpha 0.2.
5. Prints the final ESC-50 class.

## Web demo on AWS ECS

The Gradio app accepts WAV uploads and displays the predicted class with the
group probabilities and specialist fusion breakdown. Build and deploy it to
Amazon ECS Express Mode using the steps in [deploy/aws/README.md](deploy/aws/README.md).

Live demo: [ESC-50 Sound Classifier](https://es-d3438eb0c9d640c38cbe34d106f62a87.ecs.ap-southeast-2.on.aws)

To run the web demo locally with the project virtual environment, install the
web dependencies into `venv` and launch Uvicorn:

```powershell
.\venv\Scripts\python.exe -m pip install gradio fastapi "uvicorn[standard]"
.\venv\Scripts\python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8080
```

Then open `http://127.0.0.1:8080` in a browser.

## Model checkpoints

The `models/` directory contains the final eight trained checkpoints.

## Training

`src/training/train_utils.py` contains reusable training and checkpoint
utilities. `run_training.py` documents the training entry point.

The complete original training experiment is not automatically rerun
by this package. The included checkpoints are the trained final models.

## Final experiment metrics

- Test samples: 397
- Correct predictions: 320
- End-to-end accuracy: 80.60%
- Macro-F1: 0.8034
- Weighted-F1: 0.8029
- Level-1 Top-1 routing accuracy: 82.62%
- Level-2 conditional accuracy: 97.56%

See `results/metrics/` for machine-readable metrics.

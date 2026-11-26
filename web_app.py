from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from run_inference import run_inference


CUSTOM_CSS = """
:root {
  --page: #f3f7f7;
  --ink: #142b35;
  --muted: #637982;
  --teal: #087e80;
  --line: #dce8e8;
}
body, .gradio-container { background: var(--page) !important; color: var(--ink); }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; }
#hero { padding: 30px 34px 26px; border-radius: 22px; background: linear-gradient(120deg,#102f3b,#145e62); color: white; margin-bottom: 18px; }
#hero h1 { color: white; font-size: clamp(2rem,4vw,3rem); line-height: 1.08; margin: 10px 0; letter-spacing: -.04em; }
#hero p { color: #d4e8e8; font-size: 1.03rem; max-width: 660px; margin-bottom: 0; }
#eyebrow { color: #a5e5d7; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.panel { background: white; border: 1px solid var(--line); border-radius: 18px; padding: 20px; box-shadow: 0 8px 28px rgba(28,68,76,.05); }
.panel h2 { margin-top: 0; color: var(--ink); }
#classify { min-height: 50px; border-radius: 12px; background: var(--teal); }
#classify:hover { background: #06696b; }
#prediction textarea { font-size: 1.35rem !important; font-weight: 700; color: #075f61 !important; }
#decision textarea { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .86rem; line-height: 1.5; }
#hint { color: var(--muted); font-size: .9rem; }
footer { display: none !important; }
@media (max-width: 700px) { #hero { padding: 24px 22px; } .panel { padding: 15px; } }
"""


def classify_audio(audio_path):
    if not audio_path:
        return "Upload a WAV clip to begin.", ""

    try:
        captured_output = StringIO()
        with redirect_stdout(captured_output):
            result = run_inference(Path(audio_path))
        details = captured_output.getvalue()
        # The temporary upload path is operational detail, not part of the decision.
        details = "\n".join(
            line for line in details.splitlines()
            if not line.startswith("Audio file:")
        )
        return result["class"], details
    except Exception as exc:
        return "Prediction failed", f"{type(exc).__name__}: {exc}"


with gr.Blocks(
    title="ESC-50 Sound Classifier",
    theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    css=CUSTOM_CSS,
) as demo:
    gr.HTML(
        '<section id="hero"><div id="eyebrow">Audio intelligence · ESC-50</div>'
        '<h1>What can you hear?</h1>'
        '<p>Upload an environmental sound and see which of the ESC-50 classes '
        'best matches it, along with the model’s decision path.</p></section>'
    )
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes="panel"):
            gr.Markdown("## 1 · Add an audio clip")
            audio_upload = gr.File(
                label="Choose a WAV file",
                file_types=[".wav"],
                type="filepath",
            )
            gr.Markdown(
                "WAV format · The model compares the clip across sound groups "
                "and specialist classifiers.",
                elem_id="hint",
            )
            classify_button = gr.Button(
                "Classify sound", variant="primary", elem_id="classify"
            )
        with gr.Column(scale=7, elem_classes="panel"):
            gr.Markdown("## 2 · Prediction")
            predicted_class = gr.Textbox(
                label="Predicted class", value="Your result will appear here",
                interactive=False, elem_id="prediction",
            )
            gr.Markdown("## How the model decided")
            decision_details = gr.Textbox(
                label="Probabilities, selected routes, and fused scores",
                lines=17,
                max_lines=30,
                interactive=False,
                elem_id="decision",
                placeholder="The decision breakdown appears after classification.",
            )
    classify_button.click(
        fn=classify_audio,
        inputs=audio_upload,
        outputs=[predicted_class, decision_details],
    )

app = FastAPI(title="ESC-50 Sound Classifier")


@app.get("/health")
def health_check():
    return {"status": "ok"}


app = gr.mount_gradio_app(app, demo, path="/")

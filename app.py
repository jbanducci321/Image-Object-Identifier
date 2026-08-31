"""
app.py

Gradio UI that wires together query_parser.py, detector.py, and
postprocess.py into the "AI I Spy" demo.
"""

import os

import gradio as gr
from PIL import Image, ImageDraw

from detector import run_detection
from postprocess import get_best_detection
from query_parser import extract_object_phrase

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
# Calibrated against this project's preset images: owlv2-base-patch16-ensemble
# scores true positives mostly in the 0.07-0.30 range on cluttered scenes.
CONFIDENCE_THRESHOLD = 0.07

_SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


def list_preset_images():
    if not os.path.isdir(IMAGES_DIR):
        return []
    return sorted(
        f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(_SUPPORTED_EXTENSIONS)
    )


def load_preset_image(image_filename: str):
    """Show the unannotated preset image as soon as it's picked from the dropdown."""
    if not image_filename:
        return None, ""
    image_path = os.path.join(IMAGES_DIR, image_filename)
    return Image.open(image_path), ""


def draw_box(image: Image.Image, box, label: str) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    draw.text((x1 + 4, max(y1 - 20, 0)), label, fill="red")
    return annotated


def search(image_filename: str, sentence: str):
    if not image_filename:
        return None, "Please select a preset image first."
    if not sentence or not sentence.strip():
        return None, "Please type a sentence describing what to find."

    image_path = os.path.join(IMAGES_DIR, image_filename)
    image = Image.open(image_path)

    object_phrase = extract_object_phrase(sentence)

    logits, pred_boxes = run_detection(image, object_phrase)
    detection = get_best_detection(
        logits,
        pred_boxes,
        image_width=image.width,
        image_height=image.height,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    if detection is None:
        return image, f"Couldn't find '{object_phrase}' in the image."

    annotated = draw_box(image, detection["box"], object_phrase)
    status = f"Found '{object_phrase}' at confidence {detection['score']:.2f}"
    return annotated, status


with gr.Blocks(title="AI I Spy") as demo:
    gr.Markdown("# AI I Spy")
    gr.Markdown(
        "Type a sentence describing an object in the picture, and OWL-ViT "
        "will try to find it — just like the classic I Spy game."
    )

    with gr.Row():
        with gr.Column():
            image_dropdown = gr.Dropdown(
                choices=list_preset_images(),
                label="Preset image",
                value=list_preset_images()[0] if list_preset_images() else None,
            )
            sentence_box = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. please find the red umbrella",
            )
            search_button = gr.Button("Search", variant="primary")
        with gr.Column():
            image_output = gr.Image(label="Image")
            status_output = gr.Textbox(label="Status", interactive=False)

    image_dropdown.change(
        fn=load_preset_image,
        inputs=[image_dropdown],
        outputs=[image_output, status_output],
    )
    search_button.click(
        fn=search,
        inputs=[image_dropdown, sentence_box],
        outputs=[image_output, status_output],
    )
    demo.load(
        fn=load_preset_image,
        inputs=[image_dropdown],
        outputs=[image_output, status_output],
    )

if __name__ == "__main__":
    # Hugging Face Spaces sets SPACE_ID and manages the port itself; only
    # force port 3000 for local runs.
    if os.environ.get("SPACE_ID"):
        demo.launch()
    else:
        demo.launch(server_port=3000)

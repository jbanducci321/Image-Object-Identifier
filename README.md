# AI I Spy

A local demo app that plays "I Spy": type a sentence describing an object,
and the app tries to find it in a preset cluttered image, drawing a box
around the best match.

## How it works, and who does what

This project pairs Hugging Face's `transformers` library with raw PyTorch,
and each does a distinct job:

- **Hugging Face `transformers`** supplies the pretrained model: a
  zero-shot object detector that can look for *any* text description, not
  just a fixed list of classes it was trained on. We just load it and run
  a forward pass — no training involved.
- **PyTorch** does everything *after* the model runs. The model's raw
  output is just two tensors (per-box confidence scores and per-box
  coordinates). Turning that into "here's one box on the image, or nothing"
  is hand-written tensor math: box format conversion, confidence
  thresholding, and non-max suppression — all done manually instead of
  using the library's built-in post-processing helper.

## Files

- **`query_parser.py`** — Turns a full sentence ("please find the red
  umbrella") into a short object phrase ("red umbrella") that the model
  expects, by stripping common lead-in phrases. Plain string matching, no
  model involved.
- **`detector.py`** — Loads the pretrained model and processor from Hugging
  Face and runs inference on an (image, text) pair. Returns the raw output
  tensors (`logits`, `pred_boxes`) untouched.
- **`postprocess.py`** — Pure PyTorch. Converts predicted boxes from
  (center_x, center_y, width, height) to (x1, y1, x2, y2), scales them to
  the image's actual pixel size, applies a sigmoid + confidence threshold,
  and runs non-max suppression (written from scratch, not
  `torchvision.ops.nms`) to collapse duplicate overlapping boxes down to
  the single best detection.
- **`app.py`** — The Gradio UI. Lets you pick a preset image and type a
  sentence, wires the three modules above together, draws the winning box
  on the image, and shows a status message.
- **`images/`** — Preset images to search (`ispy1.jpg`–`ispy4.jpg`). No
  upload support yet — this is a fixed local set for now.

## A note on the model

The requirements originally called for `google/owlvit-base-patch32`
(OWL-ViT). While building this, that model turned out to produce very low
confidence scores on these cluttered, many-small-objects images — mostly
0.02–0.05 even for objects clearly visible in the frame — which made any
reasonable confidence threshold effectively unusable (verified against the
library's own reference post-processing to rule out a bug in the custom
code here).

Instead, this app uses **`google/owlv2-base-patch16-ensemble`** (OWL-ViT2),
Google's improved successor. It's still a zero-shot text-conditioned
detector with the same API shape (`Owlv2Processor` /
`Owlv2ForObjectDetection` in place of `OwlViTProcessor` /
`OwlViTForObjectDetection`), but scores true positives meaningfully higher
(commonly 0.07–0.30) on these images, so the confidence threshold in
`app.py` (`CONFIDENCE_THRESHOLD = 0.07`) actually does useful work.

This is still a zero-shot model asked to find small objects in dense
clutter — it won't get everything, and it will occasionally report a
confident-sounding match for an object that isn't really there. That's an
inherent limitation of the model, not a bug in this code.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The first run will download the model weights from Hugging Face (a few
hundred MB) and cache them locally.

## Run

```bash
python app.py
```

This launches a local Gradio server and opens the app in your browser
(default: http://127.0.0.1:7860).

## Usage

1. Pick a preset image from the dropdown.
2. Type a sentence like "please find the red umbrella" or "where is the
   frog".
3. Click Search.
4. If a match clears the confidence threshold, you'll see the image with a
   red box around the best match and a status like `Found 'frog' at
   confidence 0.18`. Otherwise you'll see `Couldn't find that in the
   image.`

## Future: hosting on Hugging Face Spaces

This app is deliberately kept to local files and a single `gr.Blocks` app
with no external services, which maps directly onto a Hugging Face Space
(Gradio SDK) later — `app.py` and `requirements.txt` are already in the
shape Spaces expects. Image upload support would be a natural next step
before hosting publicly.

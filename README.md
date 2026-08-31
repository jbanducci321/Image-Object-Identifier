---
title: AI I Spy
emoji: 🔍
colorFrom: yellow
colorTo: orange
sdk: gradio
app_file: app.py
pinned: false
---

# Zero-Shot Detection I Spy

A local demo app that plays "I Spy": type a sentence describing an object,
and the app tries to find it in a preset cluttered image, drawing a box
around the best match.

## How it works, and who does what

### Zero-shot detection, in one paragraph

A normal object detector is trained on a fixed list of classes ("dog",
"car", "person" — maybe 80 of them). Zero-shot detection means the model
was never given a fixed list — instead, at *inference* time you hand it an
arbitrary text phrase ("red umbrella") and it tells you where in the image
that phrase applies. It can do this because it was trained to compare
images and text in a shared mathematical space, not to memorize class
labels.

### Hugging Face vs. PyTorch

This project pairs Hugging Face's `transformers` library with raw PyTorch,
and each does a distinct job:

- **Hugging Face `transformers`** supplies the pretrained *brain* — the
  model architecture code plus millions of learned weights, already
  trained by Google on huge image/text datasets — and the `Owlv2Processor`,
  which handles the fiddly input prep (resizing/normalizing the image,
  tokenizing the text) so the tensors going into the model are shaped
  correctly. We just load it and run a forward pass — no training
  involved.
- **PyTorch** is the tensor math underneath everything — both *inside* the
  HF model (attention, convolutions, etc., which we don't touch) and,
  deliberately in this project, in the code written by hand in
  `postprocess.py` to turn raw numbers into a real answer: box format
  conversion, confidence thresholding via sigmoid, and non-max suppression
  — all done manually instead of calling the library's built-in
  post-processing helper.

### Inside OWL-ViT: how the encoders work

OWL-ViT is built on **CLIP**'s idea: two separate transformer encoders —
one for images, one for text — trained so that matching image/text pairs
land close together in the same embedding space, and non-matching pairs
land far apart.

- **Vision transformer (ViT) encoder**: the image is chopped into a grid
  of fixed-size patches (e.g. 16x16 pixels each — that's the "patch16" in
  the model name). Each patch is flattened into a vector, and all patches
  are fed through stacked self-attention layers, the same way words are
  processed in a language transformer. Self-attention lets every patch
  "look at" every other patch, so a patch representing part of a
  nutcracker's hat can be informed by nearby patches showing its body. The
  output is one embedding vector *per patch*, not just one for the whole
  image.
- **Text encoder**: the object phrase is tokenized and pushed through its
  own transformer, producing a single embedding vector for the phrase.

Standard CLIP would then squash the whole image down to one vector and
compare it to the text vector. OWL-ViT's key trick is that it **doesn't**
collapse the image into one vector — it keeps every patch's embedding
separate, and treats every patch as a candidate "is there an object here
matching this text?" query. Each patch embedding gets compared against the
text embedding to produce a matching score, and a small regression head
predicts a bounding box for that patch. That's why the model's raw output
is one score and one box *per patch location*, not just one answer for the
whole image — hundreds of candidate boxes come out, most of them junk, and
it's `postprocess.py`'s job to whittle that down to the real one.

### The pipeline, step by step

1. **`query_parser.py`** — pure string matching, no ML at all. OWL-ViT's
   text encoder was trained on short phrases like "a photo of a red
   umbrella," not conversational sentences, so `"please find the red
   umbrella"` gets stripped down to `"red umbrella"` before it ever
   touches the model.
2. **`detector.py`** (`run_detection()`):
   - The `Owlv2Processor` turns the image into a normalized pixel tensor
     and the text phrase into token IDs.
   - `model(**inputs)` runs the forward pass — image through the ViT
     encoder, text through the text encoder, patch embeddings compared
     against the text embedding, boxes regressed.
   - We deliberately grab the *raw* outputs: `logits` (one raw score per
     patch — how well that patch matches "red umbrella") and `pred_boxes`
     (one box per patch, in normalized `center_x, center_y, width, height`
     form, values between 0 and 1 regardless of actual image size).
3. **`postprocess.py`** — the hand-written PyTorch:
   - `torch.sigmoid(logits)` — sigmoid, not softmax, because this isn't
     "pick one class out of a fixed list" (which is what softmax is for).
     Each patch/box gets an *independent* yes/no confidence for "does this
     match the text," since the vocabulary is open-ended.
   - `confidence_threshold` filtering throws out the hundreds of
     low-confidence junk boxes.
   - `cxcywh_to_xyxy` converts center/width/height format into corner
     coordinates (top-left, bottom-right), which is what's needed to
     actually draw a rectangle.
   - `scale_to_image` multiplies the normalized 0-1 coordinates by the
     real image's pixel width/height, since the model has no idea what
     size the actual image is.
   - **NMS from scratch**: even after thresholding, you often get several
     overlapping boxes all pointing at the *same* object (neighboring
     patches all fire on the same nutcracker). NMS's job: sort boxes by
     confidence, greedily keep the best one, then throw away any remaining
     box that overlaps it too much (measured by IoU — intersection area
     over union area, in `box_iou`), repeat. What survives is one box per
     distinct object.
   - `get_best_detection` glues all of that together and returns the
     single highest-confidence surviving box, or `None`.
4. **`app.py`** just wires the UI to this pipeline and draws the winning
   box with Pillow if one exists, otherwise shows the "couldn't find it"
   message.

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

This launches a local Gradio server and opens the app in your browser at
http://127.0.0.1:3000.

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

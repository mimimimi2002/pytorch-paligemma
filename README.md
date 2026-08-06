# PaliGemma from scratch — read, annotated, and probed

PaliGemma (SigLIP vision encoder + linear projector + Gemma 2B) built from plain PyTorch,
with the real `google/paligemma-3b-pt-224` weights loaded into it. No `transformers`
modelling code is involved — only the tokenizer, the config and the checkpoint come from
the Hub, and the network itself is the code in this repository.

> This is a fork of [hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma)
> by Umar Jamil. **The model implementation is his.** What is mine is listed below and
> spelled out in [License and attribution](#license-and-attribution).

## What I added

| | |
|---|---|
| **`<loc>` detection decoder** — [detection.py](detection.py) | `detect cat` answers with `<loc0222><loc0113>…`; this parses those tokens back into pixel coordinates and draws the boxes. Not in upstream. |
| **Hugging Face weight loading** — [`resolve_model_path()`](utils.py) | Upstream required a manually downloaded checkpoint directory. Now `MODEL_PATH` also accepts a repo id and pulls only the files this implementation reads. |
| **Architecture write-up** — [below](#how-paligemma-works) | Follows one image and one prompt end to end — pixels into 256 SigLIP patch embeddings, through the projector, merged into the text sequence, then through Gemma's attention to a token — and says what each stage actually does to them. |
| **Line-by-line annotations** | Comments through `modeling_gemma.py`, `modeling_siglip.py`, `processing_paligemma.py`, `inference.py`. |
| **Inference runs** — [below](#what-it-actually-outputs) | All three of the task prefixes Google documents, run against the real weights, with the boxes drawn and the one limit this checkpoint cannot pass. |

## What it actually outputs

All runs use `google/paligemma-3b-pt-224` on the same photo
([test_images/pic1.jpeg](test_images/pic1.jpeg)), greedy decoding (`do_sample=False`),
`max_tokens_to_generate=100`, on Apple Silicon (`mps`). Greedy decoding makes these
reproducible — the script prints `prompt + decoded`, which is what the table shows.

These are the three task prefixes Google documents across the
[model card](https://huggingface.co/google/paligemma-3b-pt-224) and the
[big_vision README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md):
`caption {lang}`, `detect {things}`, `segment {things}`. There is no fourth — this is the
whole documented interface of a `pt` checkpoint.

| prompt | printed output |
|---|---|
| `caption en` | `caption en white and brown cat on the floor` |
| `detect cat` | `detect cat <loc0222><loc0113><loc0933><loc0824> cat` |
| `segment cat` | `segment cat <loc0205><loc0127><loc0889><loc0824> <seg087><seg041>…<seg055> cat` |


### Detection

<img width="500" height="auto" alt="pic1_detected" src="https://github.com/user-attachments/assets/061c07a8-563c-498f-b2b9-db9d6dff4eb2" />


`<loc0222><loc0113><loc0933><loc0824>` is `y_min, x_min, y_max, x_max` quantised to 1024
bins. Turning that into pixels is a plain linear rescale, and the reason is worth stating:
[`process_images`](processing_paligemma.py) squashes every input to a square with
`resize()`, which does **not** preserve the aspect ratio and adds no letterbox padding. So
bin 0 is always the top (or left) edge of the *original* image and bin 1023 the bottom (or
right) edge — no un-padding step is needed. [detection.py](detection.py) does the
conversion and the drawing.

### Segmentation stops one step short

`segment cat` answers in the same shape, with sixteen codewords inserted before the label:

```
<loc0205><loc0127><loc0889><loc0824> <seg087><seg041>…<seg055> cat
```

The box is recoverable — those are ordinary `<loc>` tokens, so [detection.py](detection.py)
draws them exactly as it draws a `detect` answer. The mask is not. Google describes the
`<seg>` entries as "codewords used by a lightweight referring-expression segmentation
**vector-quantized variational auto-encoder**", and that VQ-VAE decoder ships separately from
this checkpoint. So the asymmetry is real and worth stating plainly: `<loc>` is a **number**
and needs nothing to interpret it, while `<seg>` is an **index into a codebook this
repository does not have**. Same vocabulary, same softmax, one of them decodable here.

## Usage

```bash
pip install -r requirements.txt

# The PaliGemma weights are gated: accept the Gemma Terms of Use on the model
# page (https://huggingface.co/google/paligemma-3b-pt-224), then authenticate.
huggingface-cli login          # or: export HF_TOKEN=hf_...

./launch_inference.sh
```

`MODEL_PATH` in [launch_inference.sh](launch_inference.sh) accepts either a Hugging Face repo id
(`google/paligemma-3b-pt-224`, downloaded to `~/.cache/huggingface` on first run and reused
afterwards) or a path to a local directory that already contains the checkpoint. Only the
weights, `config.json` and the tokenizer are pulled from the Hub — the model itself is built
from the code in this repository.

`--output_file` is only meaningful for a `detect` prompt: the generated text is scanned for
`<loc>` tokens and, if any are found, the boxed image is written there.

# How PaliGemma works
<img width="600" height="auto" alt="PaliGemma architecture" src="https://github.com/user-attachments/assets/4c173634-3a15-4cc7-8f48-2867841bbd3d" />

PaliGemma is a vision language model which consists of a vision encoder (SigLIP), a projector, and an LLM (Gemma).

The key idea is : the image is turned into 256 vectors that live in the same space as word embeddings. From Gemma's point of view there is no "image", just a sequence of embeddings, some of which happen to have come from a picture.


## SigLIP
SigLIP is a vision encoder which takes an image and converts it into image tokens. The image is divided into small patches and each patch is converted into an embedding.

SigLIP improves on CLIP by replacing the **softmax-based contrastive loss with a pairwise sigmoid loss**. CLIP normalizes similarity scores across the entire batch, which requires gathering results from every device; SigLIP scores each image-text pair independently, so no global normalization is needed and training scales to much larger batches.

## SigLIP architecture

Every class in this diagram lives in [modeling_siglip.py](modeling_siglip.py).

```mermaid
flowchart TB
  A["pixel_values — [B, 3, 224, 224]"]
  B["patch_embedding<br/>Conv2d(3 → 1152, kernel=14, stride=14, padding=valid)"]
  C["[B, 1152, 16, 16]<br/>a 16×16 grid of non-overlapping patches"]
  D["flatten(2) then transpose(1, 2)<br/>[B, 256, 1152]"]
  E["+ position_embedding<br/>learned Embedding(256, 1152)"]
  A --> B --> C --> D --> E --> G

  subgraph ENC ["SiglipEncoderLayer × 27"]
    direction TB
    G["LayerNorm"] --> H["SiglipAttention — 16 heads × 72 dims<br/>no mask: every patch sees every patch"]
    H --> I["+ residual"]
    I --> J["LayerNorm"]
    J --> K["SiglipMLP — 1152 → 4304 → 1152, GELU(tanh)"]
    K --> M["+ residual"]
  end

  M --> N["post_layernorm"]
  N --> O["[B, 256, 1152]<br/>→ multi_modal_projector"]
```

### Step1 Divide image into small patches
A 224x224 image is cut into **14x14 pixel patches**, giving a **16x16 = 256 patch** grid — `(224 // 14) ** 2 = 256`, computed in `SiglipVisionEmbeddings`.

Patch extraction is a strided convolution, `Conv2d(kernel_size=14, stride=14, padding="valid")`. Because stride equals kernel size, the patches tile the image exactly and never overlap.

### Step2 embedding
The same convolution also does the embedding: `out_channels=1152` means each patch — 14 x 14 x 3 = 588 numbers — comes out as one 1152-dimensional vector. That is a plain linear projection of the flattened patch; the convolution is just an efficient way to apply the same projection to all 256 patches at once.

The result is then reshaped in two steps:

```
[B, 1152, 16, 16]  --flatten(2)-->  [B, 1152, 256]  --transpose(1,2)-->  [B, 256, 1152]
```

From here on the 2D layout is gone. The tensor is a **sequence of 256 vectors**, and nothing downstream knows which patch sat next to which.

### Step3 positional embedding
That lost geometry is restored by adding a **learned** table, `nn.Embedding(256, 1152)` — not the sinusoidal encodings of the original Transformer. Row *i* is simply added to patch *i*, and since patch *i* is always the same location in the grid, the model learns what it needs about spatial arrangement.

This table is also why resolution is baked into the weights rather than being a preprocessing choice. A 448x448 image would produce 1024 patches and index past the end of a 256-row table, which is why Google ships separate 224 / 448 / 896 checkpoints instead of one resolution-agnostic model.

### Step4 transformer block
`SiglipEncoderLayer`, repeated 27 times. It is **pre-norm**: the LayerNorm sits inside each residual branch, so the residual stream itself is never normalised.

```
h = h + Attention(LayerNorm(h))
h = h + MLP(LayerNorm(h))
```

Two details are visible only in the code:

- `SiglipAttention.forward` takes **no `attention_mask` argument**. There is nothing to mask — all 256 patches attend to all 256 patches, in every layer. Compare this with Gemma, where masking is the entire story.
- `SiglipMLP` is an ordinary two-layer feed-forward, 1152 → 4304 → 1152 with `gelu(approximate="tanh")` — not the gated GeGLU that Gemma uses.

### Step5 output norm
One final `post_layernorm` after all 27 layers, and the output is `[B, 256, 1152]`.

## Projection
SigLIP outputs 1152-dimensional vectors, but Gemma's hidden size is 2048. The projector is a single `nn.Linear(1152, 2048)` that resizes the **embedding dimension** so the image features can sit in the same space as text embeddings. The number of tokens (256) is unchanged.

## Gemma model
Gemma is the language model that consumes the merged sequence and generates text.

### Input layout
The processor builds one flat string and tokenizes it:

```python
f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
```

```
input_ids = [<image> x 256, <bos>, prompt tokens..., \n]
```

The `<image>` entries are pure **placeholders**. They reserve 256 slots in the sequence; their embeddings carry no visual information and are thrown away during the merge.

### Merging text and image
`PaliGemmaForConditionalGeneration.forward` does this in three steps:

1. `get_input_embeddings()(input_ids)` — embed the whole sequence, including the meaningless `<image>` rows. This allocates a correctly shaped buffer.
2. `vision_tower` + `multi_modal_projector` — produce `[B, 256, 2048]` from the pixels.
3. `_merge_input_ids_with_image_features` — use `input_ids == image_token_index` as a mask and `masked_scatter` the image features into those 256 positions.

Text and image only become a single tensor at step 3, and they only actually *interact* one step later, inside the transformer's self-attention.

One subtlety: the image features are divided by `sqrt(hidden_size)` before being scattered in. Gemma multiplies all input embeddings by `sqrt(hidden_size)` at the top of its forward pass, so this division cancels it out and keeps the image features at their original scale.

### Attention
PaliGemma is a **prefix-LM**, not a plain causal LM. The image tokens and the prompt form a prefix that attends **bidirectionally** — every image patch can see every other patch and the prompt, and vice versa. Only the generated suffix is causally masked. In this repo the prefill mask is filled with zeros (no masking at all), which implements exactly that, and assumes no padding.

### Vocabulary
PaliGemma extends the Gemma tokenizer so that detection and segmentation can be expressed as plain text generation:

| ID | tokens | count |
|---|---|---|
| 0 - 255999 | original Gemma vocabulary | 256000 |
| 256000 - 257023 | `<loc0000>` - `<loc1023>` | 1024 |
| 257024 - 257151 | `<seg000>` - `<seg127>` | 128 |
| **257152** | **`<image>`** | 1 |
| 257153 - 257215 | unused padding | 63 |
| | **vocab_size** | **257216** |

The `<loc>` tokens let the model answer "detect cat" by emitting `<loc0512><loc0256>...cat`, i.e. bounding boxes come out of the same softmax as words — [the detection run above](#detection) is this table in action. The trailing padding rounds the embedding matrix to a multiple of 64 for hardware efficiency.

Because these tokens are registered with `tokenizer.add_tokens()` rather than as *special* tokens, `decode(..., skip_special_tokens=True)` leaves them in the output — which is why `detect` results are readable at all without changing the decoding call.

The embedding table is **tied** between input and output (`embed_tokens` and `lm_head` share weights), so those 257216 x 2048 ≈ 527M parameters are only stored once.

### Feed-forward
Each layer's MLP uses **GeGLU**: two parallel projections `2048 -> 16384`, one passed through GELU and multiplied elementwise with the other, then projected back `16384 -> 2048`. With `num_key_value_heads: 1` (aggressive GQA) shrinking the attention blocks, the MLPs hold most of Gemma's parameters — roughly 100M per layer x 18 layers.


## Open questions I have not settled

Claims above that are read off the code but not yet demonstrated by an experiment:

- **The prefix-LM claim.** The *design* is confirmed by Google's own wording — image and prefix tokens go through the decoder with "full block-attention", the suffix with "masked attention" — and [the prefill mask](modeling_gemma.py) being all zeros implements it. What is untested is whether it *matters*: replacing it with a triangular mask should visibly degrade captions if bidirectional prefix attention carries the weight the paper implies.
- **KV-cache equivalence.** Because the prefill branch masks nothing, running *without* a cache would let already-generated tokens attend to each other bidirectionally, so on-cache and off-cache generation should **not** agree. A correct prefix-LM mask (bidirectional prefix, causal suffix) looks like a prerequisite for that equivalence, not an optimisation.
- **Parameter accounting.** The 527M embedding and "MLPs hold most of the parameters" figures are computed by hand from the config, not measured from the loaded `state_dict`.

## License and attribution

This repository is a **derivative work**, and the parts have different owners.

| | Origin | Terms |
|---|---|---|
| `inference.py`, `modeling_gemma.py`, `modeling_siglip.py`, `processing_paligemma.py`, `utils.py`, `launch_inference.sh`, `notes/` | [hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma) by Umar Jamil | **No license published upstream** — all rights reserved by the original authors. Reproduced here for study only. |
| `detection.py`, `resolve_model_path()` in `utils.py`, this README and the explanatory code comments | Mine | MIT — see [LICENSE](LICENSE) |
| PaliGemma weights | Google | [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Not included here; download separately. |

Because the upstream repository carries no license, the base implementation is
technically "all rights reserved" and this repository cannot grant you any
rights to it. If you want to build on the model code itself, ask upstream to
add a license first.

The [LICENSE](LICENSE) file states this scope explicitly and covers only my own
contributions.

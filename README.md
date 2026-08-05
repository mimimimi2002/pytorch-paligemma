# pytorch-paligemma
This is a replication of the pytorch-paligemma(https://github.com/hkproj/pytorch-paligemma)

The model code is Umar Jamil's; the notes below are mine. See [License and attribution](#license-and-attribution).

# Paligemma structure
<img width="600" height="auto" alt="スクリーンショット 2026-07-27 16 13 08" src="https://github.com/user-attachments/assets/4c173634-3a15-4cc7-8f48-2867841bbd3d" />

PaliGemma is a vision language model which consists of a vision encoder (SigLIP), a projector, and an LLM (Gemma).

The key idea is : the image is turned into 256 vectors that live in the same space as word embeddings. From Gemma's point of view there is no "image", just a sequence of embeddings, some of which happen to have come from a picture.


## SigLIP
SigLIP is a vision encoder which takes an image and converts it into image tokens. The image is divided into small patches and each patch is converted into an embedding.

SigLIP improves on CLIP by replacing the **softmax-based contrastive loss with a pairwise sigmoid loss**. CLIP normalizes similarity scores across the entire batch, which requires gathering results from every device; SigLIP scores each image-text pair independently, so no global normalization is needed and training scales to much larger batches.

In this model we take a 224x224 image and divide it into **14x14 pixel patches**, giving a **16x16 = 256 patch** grid:

```
num_patches = (image_size // patch_size) ** 2 = (224 // 14) ** 2 = 256
```

Patch extraction is just a strided convolution — `Conv2d(kernel_size=14, stride=14)` in `modeling_siglip.py` — so the patches never overlap. The output is `[Batch, 256, 1152]`.

Note that `image_size` is baked into the weights, not just a preprocessing option: the position embedding is a learned `nn.Embedding(256, 1152)` table. Feeding a 448x448 image would produce 1024 patches and overflow that table, which is why Google ships separate 224 / 448 / 896 checkpoints instead of one resolution-agnostic model.

| variant | grid | image tokens |
|---|---|---|
| 224 | 16x16 | 256 |
| 448 | 32x32 | 1024 |
| 896 | 64x64 | 4096 |

This encoder is already pretrained on images and their corresponding captions using contrastive learning, and stays frozen in spirit — PaliGemma simply reuses it.

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

The `<loc>` tokens let the model answer "detect cat" by emitting `<loc0512><loc0256>...cat`, i.e. bounding boxes come out of the same softmax as words. The trailing padding rounds the embedding matrix to a multiple of 64 for hardware efficiency.

The embedding table is **tied** between input and output (`embed_tokens` and `lm_head` share weights), so those 257216 x 2048 ≈ 527M parameters are only stored once.

### Feed-forward
Each layer's MLP uses **GeGLU**: two parallel projections `2048 -> 16384`, one passed through GELU and multiplied elementwise with the other, then projected back `16384 -> 2048`. With `num_key_value_heads: 1` (aggressive GQA) shrinking the attention blocks, the MLPs hold most of Gemma's parameters — roughly 100M per layer x 18 layers.

## Config gotcha
The default arguments in `PaliGemmaConfig.__init__` (`image_token_index=256000`, `vocab_size=257152`) and in `SiglipVisionConfig` (`patch_size=16`) **do not match the real checkpoint**. They are inherited from HuggingFace's class signatures and are always overwritten by `config.json`:

```python
config = PaliGemmaConfig(**model_config_file)   # utils.py
```

The actual values are `image_token_index=257152`, `vocab_size=257216`, `patch_size=14`. Read `config.json`, not the defaults.

## License and attribution

This repository is a **derivative work**, and the parts have different owners.

| | Origin | Terms |
|---|---|---|
| `inference.py`, `modeling_gemma.py`, `modeling_siglip.py`, `processing_paligemma.py`, `utils.py`, `launch_inference.sh`, `notes/` | [hkproj/pytorch-paligemma](https://github.com/hkproj/pytorch-paligemma) by Umar Jamil | **No license published upstream** — all rights reserved by the original authors. Reproduced here for study only. |
| This README and the explanatory code comments | Mine | MIT — see [LICENSE](LICENSE) |
| PaliGemma weights | Google | [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Not included here; download separately. |

Because the upstream repository carries no license, the base implementation is
technically "all rights reserved" and this repository cannot grant you any
rights to it. If you want to build on the model code itself, ask upstream to
add a license first.

The [LICENSE](LICENSE) file states this scope explicitly and covers only my own
contributions.

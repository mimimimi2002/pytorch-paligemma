from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def resolve_model_path(model_path: str) -> str:
    """Return a local directory holding the model files.

    If `model_path` is an existing local directory it is used as is; otherwise it
    is treated as a Hugging Face repo id (e.g. "google/paligemma-3b-pt-224") and
    downloaded to the local HF cache (~/.cache/huggingface, or $HF_HOME).
    Gated repos such as PaliGemma need `huggingface-cli login` or $HF_TOKEN.
    """
    if os.path.isdir(model_path):
        return model_path

    print(f"Downloading {model_path} from the Hugging Face Hub (cached after the first run)")
    return snapshot_download(
        repo_id=model_path,
        # Only the files this implementation reads: weights, config and tokenizer.
        allow_patterns=["*.safetensors", "*.json", "*.model"],
    )


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    model_path = resolve_model_path(model_path)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right"

    # Load model weights from the safetensors files
    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the state dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie weights
    model.tie_weights()

    return (model, tokenizer)
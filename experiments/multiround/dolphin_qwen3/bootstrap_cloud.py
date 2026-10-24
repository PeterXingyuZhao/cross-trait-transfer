import subprocess
import sys

# Start with a fresh A100 runtime. Do not run setup over an active model.
if any(name in sys.modules for name in ("torch", "transformers", "peft")):
    raise RuntimeError("Restart the runtime, then run this installation cell first.")

bootstrap = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "loguru==0.7.3"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
if bootstrap.returncode:
    raise RuntimeError(bootstrap.stdout)
from loguru import logger

def install_command(arguments):
    """Keep installation diagnostics visible in the saved notebook output."""
    with subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        for line in process.stdout:
            logger.info(line.rstrip())
        status = process.wait()
    if status:
        raise RuntimeError(f"Installation failed with exit status {status}")

# Match the native stack used by the earlier Qwen3 notebooks.
install_command([sys.executable, "-m", "pip", "install", "-q",
    "--extra-index-url", "https://download.pytorch.org/whl/cu130",
    "torch==2.11.0+cu130", "torchvision==0.26.0+cu130", "torchaudio==2.11.0+cu130",
    "transformers==4.57.6", "peft==0.20.0", "accelerate==1.14.0",
    "tokenizers==0.22.2", "safetensors==0.7.0", "numpy==2.1.3"])
# Optional preinstalled TorchAO integrations previously broke ordinary BF16 LoRA.
install_command([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
if not torch.cuda.is_available():
    raise RuntimeError("Select an A100 GPU in Runtime > Change runtime type.")
device = torch.cuda.get_device_properties(0)
if "A100" not in device.name or device.total_memory < 38 * 2**30:
    raise RuntimeError("This BF16 notebook requires an A100 with at least 40 GB VRAM.")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("BF16 is not supported by this runtime.")
# A small CPU forward pass catches import/LoRA problems before the 8B download.
tiny = get_peft_model(Qwen3ForCausalLM(Qwen3Config(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
    num_attention_heads=2, num_key_value_heads=1, pad_token_id=0,
    bos_token_id=1, eos_token_id=2)), LoraConfig(
        r=8, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
with torch.no_grad():
    assert torch.isfinite(tiny(input_ids=torch.tensor([[1, 3, 4]])).logits).all()
del tiny
logger.success("Native LoRA check passed. GPU: {}; VRAM: {:.1f} GiB",
               device.name, device.total_memory / 2**30)


from pathlib import Path

# vLLM has its own Python/Torch installation. It does not replace native Torch.
install_command([sys.executable, "-m", "pip", "install", "-q", "uv==0.8.13"])
VLLM_ENV = Path("/content/recursive_vllm_env")
VLLM_PYTHON = VLLM_ENV / "bin/python"
if not VLLM_PYTHON.exists():
    install_command(["uv", "venv", "--python", "3.11", "--seed", str(VLLM_ENV)])
install_command(["uv", "pip", "install", "--python", str(VLLM_PYTHON),
    "--torch-backend", "cu128", "vllm==0.10.0", "torch==2.7.1+cu128",
    "torchvision==0.22.1+cu128", "torchaudio==2.7.1+cu128",
    "transformers==4.53.2", "tokenizers==0.21.4", "numpy==2.2.6", "loguru==0.7.3"])
install_command([str(VLLM_PYTHON), "-c",
    "import torch, vllm; from loguru import logger; "
    "assert vllm.__version__ == '0.10.0'; "
    "assert torch.__version__ == '2.7.1+cu128'; "
    "assert torch.cuda.is_available(); logger.success('Isolated generation stack ready')"])

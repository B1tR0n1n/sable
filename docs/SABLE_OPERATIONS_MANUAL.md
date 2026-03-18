# SABLE — Complete Operations Manual

**Updated: March 15, 2026**
**Covers: Simulator, Training Pipeline, Inference, Console App, GNN Experiment**

---

## 1. FILE LOCATIONS

### Windows Desktop

| What | Where |
|------|-------|
| llama.cpp | `C:\Users\Tom\llama.cpp` |
| llama-cli.exe | `C:\Users\Tom\llama.cpp\build\bin\Release\llama-cli.exe` |
| llama-server.exe | `C:\Users\Tom\llama.cpp\build\bin\Release\llama-server.exe` |
| Nemotron GGUF (inference) | `G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf` |
| Nemotron BF16 (for GGUF merge) | `G:\models\nemotron-hf\` |
| Nemotron FP8 (training — local) | `G:\models\nemotron-fp8\` |
| Sable LoRA v1 adapter | `G:\models\sable-lora-v1\sable-lora-v1\` |
| Sable LoRA GGUF | `G:\models\sable-v1-lora.gguf` |
| Sable simulator | `C:\Users\Tom\llama.cpp\sable\sable_sim_package\` |
| Training scenarios (2K) | `C:\Users\Tom\llama.cpp\sable\sable_sim_package\data\training_large\` |
| Fog-of-war scenarios (1.2K) | `C:\Users\Tom\llama.cpp\sable\sable_sim_package\data\training_fog\` |
| Final JSONL corpus (6K) | `C:\Users\Tom\llama.cpp\sable\sable_sim_package\data\final\sable_train.jsonl` |
| SABLE Console (Electron app) | Wherever you extracted `sable-console.tar.gz` |
| GNN training script | `train_gnn.py` (download from Claude chat) |

### Docker Container (ml-jupyter)

| What | Where |
|------|-------|
| Bind mount to G: drive | `/models/` |
| Training data | `/workspace/sable_train.jsonl` |
| FP8 model (via bind mount) | `/models/nemotron-fp8/` |
| LoRA adapter (via bind mount) | `/models/sable-lora-v1/` |
| HuggingFace cache | `/models/huggingface/` |
| Jupyter Lab | `localhost:8889` (mapped from container 8888) |

---

## 2. RUNNING BASE NEMOTRON (INFERENCE)

### Interactive chat

```powershell
cd C:\Users\Tom\llama.cpp
.\build\bin\Release\llama-cli.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --jinja
```

### With LoRA adapter (fine-tuned Sable v1)

```powershell
cd C:\Users\Tom\llama.cpp
.\build\bin\Release\llama-cli.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --lora G:\models\sable-v1-lora.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --jinja
```

### API server (for SABLE Console app)

```powershell
cd C:\Users\Tom\llama.cpp
.\build\bin\Release\llama-server.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --port 8080 --jinja
```

- OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1/chat/completions`
- Health check: `curl http://127.0.0.1:8080/health`

### API server with LoRA

```powershell
cd C:\Users\Tom\llama.cpp
.\build\bin\Release\llama-server.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --lora G:\models\sable-v1-lora.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --port 8080 --jinja
```

### Performance expectations

- Prompt processing: 92+ t/s (basic), 1500+ t/s (cached)
- Generation: 95–102 t/s
- VRAM: ~17GB on GPU 0 (5090), GPU 1 idle
- Thermals: 5090 at ~35°C under inference

---

## 3. SABLE CONSOLE APP

### First-time setup

Requires Node.js installed (https://nodejs.org/).

```powershell
cd <wherever you extracted sable-console>
npm install
```

### Run in dev mode

```powershell
npm start
```

### Build portable .exe

```powershell
npm run build
```

Creates `dist/SABLE Console.exe` — standalone, no Node.js needed to run.

### Using the app

1. Start llama-server (see Section 2) in a separate terminal
2. Launch the app (`npm start` or the built `.exe`)
3. Select a topology: Small Office, Enterprise Campus, VDI Environment, Healthcare Network
4. Set a seed number and click GENERATE
5. Click any node in the graph to inspect it
6. Click **INJECT FAILURE** to simulate a hardware failure on that node
7. Press **▶** to watch the cascade propagate tick by tick
8. Scrub the timeline slider to jump to any tick
9. When cascade finishes, click **ASK NEMOTRON** for AI root cause analysis
10. Click **RESET** to start over with a new scenario
11. Click **⚙** to change the LLM endpoint if needed

### Troubleshooting

- **"Connection failed: Failed to fetch"** — llama-server isn't running or wrong port. Check `http://127.0.0.1:8080/health` in a browser.
- **App won't start** — Run `npm install` first. Make sure Node.js is installed.
- **No cascade after inject** — Some nodes have no dependents. Try injecting on a core switch or physical server for bigger cascades.

---

## 4. SABLE SIMULATOR

### Install

```powershell
cd C:\Users\Tom\llama.cpp\sable\sable_sim_package
pip install -e .
```

### Run a single scenario

```powershell
python run_scenario.py --template small_office --print-trace --seed 42
```

Templates: `small_office`, `enterprise_campus`, `vdi_environment`, `healthcare_network`

### Run with specific failure type

```powershell
python run_scenario.py --template enterprise_campus --failure CASCADING_OVERLOAD --difficulty hard --print-trace --seed 100
```

Failure categories:
- `SINGLE_COMPONENT_FAILURE`
- `CASCADING_OVERLOAD`
- `SILENT_DEGRADATION`
- `NETWORK_PARTITION`
- `DEPENDENCY_CHAIN`
- `CORRELATED_FAILURE`
- `INTERMITTENT_FAILURE`
- `CONFIGURATION_DRIFT`

### Generate training data (batch)

```powershell
# 100 scenarios, quick test
python generate_dataset.py --count 100 --output ./data/test/ --templates small_office,enterprise_campus --seed 42

# 2,000 scenarios, weighted hard
python generate_dataset.py --count 2000 --output ./data/training_large/ --templates small_office,enterprise_campus,vdi_environment,healthcare_network --difficulty-mix "easy:0.1,medium:0.4,hard:0.5" --seed 1337

# 1,200 fog-of-war focused (hard only)
python generate_dataset.py --count 1200 --output ./data/training_fog/ --templates enterprise_campus,vdi_environment,healthcare_network --difficulty-mix "hard:1.0" --seed 7777

# With GNN graph export
python generate_dataset.py --count 2000 --output ./data/gnn/ --export-pyg --templates small_office,enterprise_campus --seed 42
```

### Visualize a topology

```powershell
python visualize_topology.py --template enterprise_campus --output ./topology.png --seed 42
```

### Convert scenarios to fine-tuning JSONL

```powershell
python build_training_pairs.py --input-dir ./data/training_large/ --output-dir ./data/finetune_large/
```

### Mix training corpus

```powershell
python mix_training_corpus.py -s ./data/finetune_large/ -s ./data/finetune_fog/ -o ./data/final/sable_train.jsonl --cascade-count 1000 --fogofwar-count 3000 --decision-count 2000 --seed 42
```

### Convert to llama.cpp format

```powershell
python convert_for_llama.py -i ./data/final/sable_train.jsonl -o ./data/final/sable_train.txt
```

---

## 5. FINE-TUNING ON RUNPOD

### Files to upload

1. `sable_train.jsonl` — training corpus (from `data/final/`)
2. `runpod_setup.sh` — installs deps
3. `train_sable.py` — training script

### Pod setup

- GPU: A100 80GB or H100 SXM 80GB
- Template: RunPod PyTorch
- Container disk: leave default (20GB is fine)
- Volume: not needed for single run

### Execution

```bash
# Set temp/cache to workspace (network volume has the space)
export TMPDIR=/workspace/tmp
export PIP_CACHE_DIR=/workspace/pip-cache
export HF_HOME=/workspace/huggingface
mkdir -p $TMPDIR $PIP_CACHE_DIR $HF_HOME

# Install deps (use --no-build-isolation to avoid disk space issues)
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
pip install peft accelerate bitsandbytes datasets
pip install transformers --upgrade

# Download model (in a second terminal to parallelize with mamba build)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', local_dir='/workspace/nemotron-bf16')"

# Update training script to use local model
sed -i 's|MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"|MODEL_ID = "/workspace/nemotron-bf16"|' /workspace/train_sable.py

# Run training
python3 train_sable.py

# Package adapter for download
tar czf /workspace/sable-lora-v1.tar.gz -C /workspace sable-lora-v1/
```

### Download and kill pod

Download `sable-lora-v1.tar.gz` via RunPod file manager, then terminate the pod immediately.

### Known RunPod issues

- **Disk full during mamba-ssm build**: Set `TMPDIR=/workspace/tmp` before installing. The 20GB root partition fills up if pip builds in `/tmp`.
- **huggingface-cli not found**: Use `python3 -c "from huggingface_hub import snapshot_download; ..."` instead.
- **Version mismatches**: The pod's pre-installed PyTorch may not match transformers. If you get `set_submodule` errors, `pip install torch --upgrade`. If you get `metadata.get` errors, the transformers version doesn't match the model format.

### Cost

H100 SXM at ~$2.39/hr. Full run including setup + model download + 4hr training ≈ $12–15. A100 80GB at ~$2/hr would be ~$10–12.

---

## 6. MERGING LORA INTO GGUF

After getting the adapter from RunPod, merge it into a GGUF for llama.cpp:

### Prerequisites

- Base BF16 model at `G:\models\nemotron-hf\` (~60GB)
- LoRA adapter at `G:\models\sable-lora-v1\sable-lora-v1\`

### Download BF16 model (if not already present)

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', local_dir='G:/models/nemotron-hf')"
```

### Convert LoRA to GGUF

```powershell
cd C:\Users\Tom\llama.cpp
python convert_lora_to_gguf.py --base G:\models\nemotron-hf\ --outfile G:\models\sable-v1-lora.gguf G:\models\sable-lora-v1\sable-lora-v1\
```

### Run with merged LoRA

```powershell
.\build\bin\Release\llama-cli.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --lora G:\models\sable-v1-lora.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --jinja
```

Note: The LoRA is applied at runtime on top of the base GGUF. The base model stays unchanged.

---

## 7. GNN EXPERIMENT (LAPTOP)

For the RTX 3070 laptop (8GB VRAM). Trains a graph neural network to predict cascade outcomes directly on the topology graph — no language model involved.

### Setup

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric numpy click tqdm scikit-learn
```

### Generate graph training data

```powershell
cd C:\Users\Tom\llama.cpp\sable\sable_sim_package
python generate_dataset.py --count 2000 --output ./data/gnn_train/ --templates small_office,enterprise_campus,vdi_environment,healthcare_network --difficulty-mix "easy:0.2,medium:0.4,hard:0.4" --seed 42
```

### Train

```powershell
python train_gnn.py --data-dir ./data/gnn_train/ --task all --epochs 100
```

Tasks:
- `node_state` — predict final state of every node after cascade
- `severity` — predict overall cascade severity (minimal/moderate/severe)
- `all` — both

VRAM usage: under 1GB. Runs fine on the 3070 laptop.

---

## 8. DOCKER ML STACK

### Start container

```powershell
docker start ml-jupyter
docker exec -it ml-jupyter bash
```

### Start Jupyter

```bash
cd /workspace
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
jupyter server list
```

Access at `http://localhost:8889`

### Copy files into container

```powershell
# From Windows PowerShell
docker cp C:\path\to\file ml-jupyter:/workspace/
docker cp "G:\models\sable-lora-v1\." ml-jupyter:/models/sable-lora-v1/
```

### Verify GPU in container

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Container details

- Image: `ml-jupyter-snapshot:2026-03-11` (74.6GB)
- PyTorch: 2.9.0a0 (compiled from source, sm_120)
- mamba-ssm: 2.3.1 (compiled from source)
- CUDA: 13.0
- Python: 3.12
- Ports: 8888→8889, 7860→7860, 7880→7880

---

## 9. GPU MONITORING

```powershell
# Quick check
nvidia-smi

# Continuous (refreshes every 2s)
nvidia-smi -l 2
```

### Hardware

| GPU | VRAM | Slot | Role |
|-----|------|------|------|
| RTX 5090 | 32GB GDDR7 | Top x16 (full bandwidth) | Primary — inference + training |
| RTX 5070 Ti | 16GB GDDR7 | Second slot (x4 chipset) | Overflow — dual-GPU inference for large models |
| Combined | 48GB | | |

---

## 10. V1 FINE-TUNE RESULTS AND LESSONS

### What happened

- Training: 6,000 examples, 1 epoch, BF16 on H100 SXM, ~4 hours
- Final loss: 0.08
- The model memorized the training data format but did not generalize well
- Base Nemotron outperforms the fine-tune on novel scenarios

### Why

- Training corpus was 100% large-topology verbose examples
- Model learned "when I see infrastructure, output 20 pages of analysis"
- Hallucinated phantom components not present in the prompt
- No concise output examples in the training mix

### What to fix for v2

1. Add small topology scenarios (small_office, minimal hand-crafted 3–5 node examples)
2. Add concise output format — 5–10 line responses, not full ground-truth dumps
3. Include system prompts that constrain output length in training pairs
4. Mix 50% concise / 50% detailed so the model learns to follow format instructions
5. Add examples where the model must say "I don't know" or "insufficient information"

### What worked

- The full pipeline is proven end-to-end: simulator → JSONL → RunPod → LoRA → GGUF → inference
- The model absorbed domain vocabulary (dependency types, cascade mechanics, tick propagation, DNS TTL)
- LoRA merge into GGUF works — runtime application via llama.cpp `--lora` flag
- The training infrastructure (RunPod, mamba-ssm compilation, BF16 model path) is documented and repeatable

---

## 11. NVIDIA BUG REPORT

### Where to post

NVIDIA Developer Forums → CUDA Setup and Installation

### Title

"RTX 5090 + 5070 Ti Multi-GPU Training: CUDA Driver Crash During Backward Pass (sm_120, PyTorch, gradient_checkpointing)"

### The two bugs

1. **With gradient_checkpointing=True**: "expected device meta but got cuda:0" error during gradient computation across device boundary
2. **Without gradient_checkpointing**: CUDA driver crashes with "device not ready", requires full system reboot. Driver should OOM gracefully, never crash.

### Before posting

Get exact driver version: `nvidia-smi` → top line shows driver version. Add to the report.

---

## 12. QUICK REFERENCE

### Start everything for a Sable session

```powershell
# Terminal 1: llama-server
cd C:\Users\Tom\llama.cpp
.\build\bin\Release\llama-server.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --port 8080 --jinja

# Terminal 2: SABLE Console
cd <sable-console directory>
npm start
```

### Generate fresh training data

```powershell
cd C:\Users\Tom\llama.cpp\sable\sable_sim_package
python generate_dataset.py --count 2000 --output ./data/new_batch/ --templates small_office,enterprise_campus,vdi_environment,healthcare_network --difficulty-mix "easy:0.2,medium:0.4,hard:0.4" --seed 9999
```

### Test a specific failure scenario

```powershell
python run_scenario.py --template healthcare_network --failure DEPENDENCY_CHAIN --difficulty hard --print-trace --seed 123
```

### Check if llama-server is alive

```powershell
curl http://127.0.0.1:8080/health
```

---

*Built by Keith. Hardware to software to model to app. The full stack.*

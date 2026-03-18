# LLAMA.CPP + DUAL-GPU INFERENCE — BUILD HANDOFF

**Updated: March 14, 2026**

---

## HARDWARE

- **GPU 0:** PNY GeForce RTX 5090 EPIC-X ARGB OC — 32GB GDDR7
- **GPU 1:** MSI GeForce RTX 5070 Ti Ventus 3X OC — 16GB GDDR7
- **Combined VRAM:** 48GB
- **CPU:** AMD Ryzen 9 9950X3D
- **RAM:** 64GB DDR5-6000
- **Motherboard:** MSI MEG X870E Godlike
- **PSU:** 1250W
- **PCIe Layout:** 5090 in top x16 slot (full bandwidth), 5070 Ti in second slot (x4 chipset)

---

## THERMAL VALIDATION

- **Stress tested:** OCCT 3D Adaptive Steady, Heavy load, 15 minutes, both GPUs
- **5090:** 69°C core / 72°C memory junction @ 338W
- **5070 Ti:** 63°C core / 62°C memory junction @ 191W
- **Combined draw:** ~530W
- **Zero PCIe lane errors, zero WHEA errors**
- **Bottom 140mm intake fan feeding both cards**
- **5070 Ti sag fixed with rubber stopper + double-sided tape**

---

## BUILD ENVIRONMENT

- **OS:** Windows (native, not WSL)
- **Visual Studio:** Community 2026
- **CUDA Toolkit:** v13.2 (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`)
- **CMake:** v4.3.0-rc3
- **Git:** v2.5
- **Python:** 3.14 (C:\Python314) + 3.13 (user install)

### PATH Notes
- Old CUDA v12.3 was on PATH — updated to v13.2 via:
  ```
  [Environment]::SetEnvironmentVariable("Path", $env:Path.Replace("CUDA\v12.3\bin", "CUDA\v13.2\bin"), "User")
  ```
- CMake added to PATH during install
- `huggingface-cli` not on PATH — use Python module invocation instead

---

## LLAMA.CPP BUILD

**Location:** `C:\Users\Tom\llama.cpp`

**Build commands:**
```powershell
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

**Build output:** `C:\Users\Tom\llama.cpp\build\bin\Release\llama-cli.exe`

**Build completed successfully — March 14, 2026**

---

## MODELS

**Storage location:** `G:\models\`

### Nemotron-3-Nano-30B-A3B (Q4_K_XL) — DOWNLOADING
- **Source:** `unsloth/Nemotron-3-Nano-30B-A3B-GGUF`
- **Quantization:** UD-Q4_K_XL (Dynamic 2.0)
- **Size:** ~18GB
- **Architecture:** MoE hybrid Mamba-Transformer, 30B total / 3B active params
- **Local path:** `G:\models\nemotron-nano\`
- **Download command:**
  ```python
  python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Nemotron-3-Nano-30B-A3B-GGUF', local_dir='G:/models/nemotron-nano', allow_patterns=['*UD-Q4_K_XL*'])"
  ```

### Qwen2.5-72B-Instruct (Q4_K_M) — PLANNED
- **Source:** TBD (bartowski or Qwen official GGUF)
- **Quantization:** Q4_K_M
- **Size:** ~42GB
- **Local path:** `G:\models\qwen2.5-72b\` (planned)
- **Rationale:** Stronger structured reasoning for Sable development work

---

## RUNNING INFERENCE — DUAL GPU

### Basic run (Nemotron Nano — fits on single GPU):
```powershell
cd C:\Users\Tom\llama.cpp
.\build\bin\Release\llama-cli.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --jinja
```
- `-ngl 999` offloads all layers to GPU (will use GPU 0 by default)

### Dual-GPU layer splitting (for larger models like Qwen 72B):
```powershell
.\build\bin\Release\llama-cli.exe -m G:\models\qwen2.5-72b\<model-file>.gguf --ctx-size 8192 --temp 0.6 --top-p 0.95 -ngl 999 --tensor-split 67,33
```
- `--tensor-split 67,33` allocates ~67% of layers to GPU 0 (5090, 32GB) and ~33% to GPU 1 (5070 Ti, 16GB)
- Adjust the ratio based on actual VRAM usage — monitor with `nvidia-smi`

### Server mode (OpenAI-compatible API):
```powershell
.\build\bin\Release\llama-server.exe -m G:\models\nemotron-nano\Nemotron-3-Nano-30B-A3B-UD-Q4_K_XL.gguf --ctx-size 16384 --temp 0.6 --top-p 0.95 -ngl 999 --port 8080 --jinja
```
- Access at `http://127.0.0.1:8080`
- OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1/chat/completions`

---

## MONITORING

```powershell
# Check both GPUs
nvidia-smi

# Continuous monitoring (refreshes every 2 seconds)
nvidia-smi -l 2

# HWiNFO64 for detailed temps, memory junction, power draw
```

---

## NEXT STEPS

- [ ] Confirm Nemotron-3-Nano download completes
- [ ] Test Nemotron-3-Nano inference on single GPU
- [ ] Download Qwen2.5-72B-Instruct Q4_K_M
- [ ] Test dual-GPU layer splitting with Qwen 72B
- [ ] Benchmark both models against Sable-relevant prompts
- [ ] Set up llama-server for persistent API endpoint
- [ ] Integrate with ML Studio Gradio workbench

---

## KNOWN ISSUES

- **Python 3.14:** Bleeding edge — `huggingface-cli` doesn't run directly. Use `python -c` with snapshot_download instead.
- **Dual Python installs:** 3.14 at `C:\Python314`, 3.13 at user AppData. Verify which has `huggingface_hub` installed with `python -m pip show huggingface_hub`.
- **Old CUDA on PATH:** v12.3 still exists at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3` — can be uninstalled to clean up.
- **5070 Ti sag:** Physical fix in place (rubber stopper). Monitor if it shifts over time.

#!/bin/bash
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

APPIMAGE_URL="https://jcycdn.com/update/SGLang-f6adb4f-kt379b9df-x86_64.AppImage"
APPIMAGE_NAME="SGLang-f6adb4f-kt379b9df-x86_64.AppImage"
MODELS_BASE="/mnt/data/models"

declare -a MODEL_LIST=(
##   "Qwen3-30B-A3B-Instruct-2507|Qwen/Qwen3-30B-A3B-Instruct-2507"
    "Qwen3.5-122B-A10B-FP8|Qwen/Qwen3.5-122B-A10B-FP8"
    "Qwen3.5-35B-A3B-FP8|Qwen/Qwen3.5-35B-A3B-FP8"
 #   "Qwen3.5-FP8|Qwen/Qwen3.5-397B-A17B-FP8"
    "Qwen3-Coder-Next|Qwen/Qwen3-Coder-Next"
    "Qwen3-Coder-Next-FP8|Qwen/Qwen3-Coder-Next-FP8"
    "MiniMax-M2.5|MiniMaxAI/MiniMax-M2.5"

)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo -e "\n==> $*"; }
ok()  { echo "    [OK] $*"; }

# ============================================================================
# Step 1: Install Python dependencies
# ============================================================================

log "Installing Python dependencies..."
pip install --quiet requests huggingface_hub 2>/dev/null \
    || pip install requests huggingface_hub
ok "requests, huggingface_hub"

# ============================================================================
# Step 2: Create model directory
# ============================================================================

log "Creating model directory: $MODELS_BASE"
mkdir -p "$MODELS_BASE"
ok "$MODELS_BASE"

# ============================================================================
# Step 3: Download AppImage
# ============================================================================

log "Downloading AppImage..."
if [ -f "$SCRIPT_DIR/$APPIMAGE_NAME" ]; then
    ok "Already exists: $APPIMAGE_NAME (skipping)"
else
    wget -c --progress=bar:force:noscroll -O "$SCRIPT_DIR/$APPIMAGE_NAME" "$APPIMAGE_URL"
    ok "Downloaded: $APPIMAGE_NAME"
fi
chmod +x "$SCRIPT_DIR/$APPIMAGE_NAME"

# ============================================================================
# Step 4: Generate benchmark_all_models.py
# ============================================================================

BENCH_SCRIPT="$SCRIPT_DIR/benchmark_all_models.py"

log "Generating benchmark script: $BENCH_SCRIPT"
cat > "$BENCH_SCRIPT" << 'PYTHON_SCRIPT_EOF'
#!/usr/bin/env python3
"""
Automated benchmark script for all models on ktransformers + sglang.

Launches sglang server for each model, runs prefill + decode benchmarks,
records results to CSV and prints a summary table matching the standard
ktransformers benchmark format.

Features:
  - Real-time server log
  - Full launch command printed for manual reproduction
  - Incremental CSV save after each model (crash-safe)
  - Ctrl+C skips current model instead of killing the script
  - Per-model unique ports to avoid conflicts
  - Hardware info collection (CPU model, ISA, memory)

Output table format:
  +-----------------+----------+----------+----------+----------+----------+
  | Prompt          | hi (2)   | 1K (969) | 2K(1930) | 4K(3846) | 8K(7678) |
  | Output length   | 10tokens |300tokens |300tokens |300tokens |300tokens |
  +-----------------+----------+----------+----------+----------+----------+
  | Prefill token/s |   13     |  105     |  102     |   88     | CUDA OOM |
  | Decode  token/s |  16.8    |  15.4    |  14.2    |  13.0    | CUDA OOM |
  +-----------------+----------+----------+----------+----------+----------+

Requires SGLang*.AppImage in the same directory as this script.

Usage:
    python benchmark_all_models.py --models-base /path                     # custom path + all_models
    python benchmark_all_models.py --models-base /path MiniMax-M2.5        # custom path + chosen_models
"""

import argparse
import glob
import subprocess
import requests
import time
import csv
import os
import signal
import sys
import threading
import traceback
import platform
import json
import re
from datetime import datetime

# ============================================================================
# Global configuration
# ============================================================================

MODELS_BASE = "/mnt/data/models"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perf")
BASE_PORT = 30001
APPIMAGE_GLOB = "SGLang*.AppImage"

# Benchmark configs: (prompt_label, approx_prompt_tokens, max_gen_tokens)
# Matches the table format: hi(2), 1K(969), 2K(1930), 4K(3846), 8K(7678)
BENCH_CONFIGS = [
    ("hi (2)",     1,    10),
    ("1K (969)",   1024,  300),
    ("2K (1930)",  2048,  300),
    ("4K (3846)",  4096,  300),
    ("8K (7678)",  8192,  300),
]

SERVER_STARTUP_TIMEOUT = 600   # seconds
REQUEST_TIMEOUT = 600          # seconds per request
INTER_MODEL_WAIT = 10          # seconds between models
WARMUP_TIMEOUT = 120           # seconds for warmup request

# Key launch params to record for reproducibility
REPRO_KEYS = [
    ("tensor-parallel-size", "TP"),
    ("kt-num-gpu-experts",   "GPU Experts"),
    ("kt-method",            "Method"),
    ("kt-cpuinfer",          "CPU Infer"),
    ("kt-threadpool-count",  "Threadpool"),
]

# Applied to every model launch to avoid known crashes
GLOBAL_ENV = {
    "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    "SGLANG_DISABLE_CUDNN_CHECK": "1",
}


# ============================================================================
# Model configurations
# ============================================================================

def get_model_configs():
    """Return ordered list of (name, served_name, args) tuples."""
    C = []

    C.append(("Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", {
        "model": f"{MODELS_BASE}/Qwen2.5-7B-Instruct",
        "kt-weight-path": f"{MODELS_BASE}/Qwen2.5-7B-Instruct",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-method": "BF16",
        "attention-backend": "triton",
        "trust-remote-code": True, "mem-fraction-static": 0.85,
        "chunked-prefill-size": 4096, "max-running-requests": 1,
        "max-total-tokens": 32000,
        "tensor-parallel-size": 1,
    }))

    C.append(("Qwen3-30B-A3B-Instruct-2507", "Qwen3-30B-A3B", {
        "model": f"{MODELS_BASE}/Qwen3-30B-A3B-Instruct-2507",
        "kt-weight-path": f"{MODELS_BASE}/Qwen3-30B-A3B-Instruct-2507",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-num-gpu-experts": 28, "kt-method": "BF16",
        "attention-backend": "triton",
        "trust-remote-code": True, "mem-fraction-static": 0.85,
        "chunked-prefill-size": 4096, "max-running-requests": 1,
        "max-total-tokens": 32000,
        "enable-mixed-chunk": True, "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
    }))

    C.append(("Qwen3.5-122B-A10B-FP8", "Qwen3.5-122B-A10B-FP8", {
        "model": f"{MODELS_BASE}/Qwen3.5-122B-A10B-FP8",
        "kt-weight-path": f"{MODELS_BASE}/Qwen3.5-122B-A10B-FP8",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-num-gpu-experts": 16, "kt-method": "FP8",
        "kt-gpu-prefill-token-threshold": 2048,
        "attention-backend": "triton",
        "trust-remote-code": True, "mem-fraction-static": 0.85,
        "chunked-prefill-size": 4096, "max-running-requests": 1,
        "max-total-tokens": 32000,
        "enable-mixed-chunk": True, "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
    }))

    C.append(("Qwen3.5-35B-A3B-FP8", "Qwen3.5-35B-A3B-FP8", {
        "model": f"{MODELS_BASE}/Qwen3.5-35B-A3B-FP8",
        "kt-weight-path": f"{MODELS_BASE}/Qwen3.5-35B-A3B-FP8",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-num-gpu-experts": 32, "kt-method": "FP8",
        "kt-gpu-prefill-token-threshold": 400,
        "attention-backend": "triton",
        "trust-remote-code": True, "mem-fraction-static": 0.85,
        "chunked-prefill-size": 4096, "max-running-requests": 1,
        "max-total-tokens": 32000,
        "enable-mixed-chunk": True, "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
    }))


    C.append(("Qwen3.5-FP8", "Qwen3.5-FP8", {
        "model": f"{MODELS_BASE}/Qwen3.5-FP8",
        "kt-weight-path": f"{MODELS_BASE}/Qwen3.5-FP8",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-num-gpu-experts": 4, "kt-method": "FP8",
        "kt-gpu-prefill-token-threshold": 2048,
        "attention-backend": "triton",
        "trust-remote-code": True, "mem-fraction-static": 0.85,
        "chunked-prefill-size": 4096, "max-running-requests": 1,
        "max-total-tokens": 32000,
        "enable-mixed-chunk": True, "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
    }))

    C.append(("Qwen3-Coder-Next", "Qwen3-Coder-Next", {
        "model": f"{MODELS_BASE}/Qwen3-Coder-Next",
        "kt-weight-path": f"{MODELS_BASE}/Qwen3-Coder-Next",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-num-gpu-experts": 6, "kt-method": "BF16",
        "kt-gpu-prefill-token-threshold": 2048,
        "attention-backend": "triton",
        "trust-remote-code": True, "mem-fraction-static": 0.80,
        "chunked-prefill-size": 16384, "max-running-requests": 1,
        "max-total-tokens": 64000,
        "enable-mixed-chunk": True, "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
        "kt-enable-dynamic-expert-update": True,
    }))

    C.append(("Qwen3-Coder-Next-FP8", "Qwen3-Coder-Next-FP8", {
        "model": f"{MODELS_BASE}/Qwen3-Coder-Next-FP8",
        "kt-weight-path": f"{MODELS_BASE}/Qwen3-Coder-Next-FP8",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-num-gpu-experts": 16, "kt-method": "FP8",
        "kt-gpu-prefill-token-threshold": 2048,
        "attention-backend": "triton",
        "trust-remote-code": True, "mem-fraction-static": 0.80,
        "chunked-prefill-size": 16384, "max-running-requests": 1,
        "max-total-tokens": 64000,
        "enable-mixed-chunk": True, "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
        "kt-enable-dynamic-expert-update": True,
    }))

    # 230B MoE (256 experts, 8 active), FP8 weights, Lightning Attention
    C.append(("MiniMax-M2.5", "MiniMax-M2.5", {
        "model": f"{MODELS_BASE}/MiniMax-M2.5",
        "kt-weight-path": f"{MODELS_BASE}/MiniMax-M2.5",
        "kt-cpuinfer": 16, "kt-threadpool-count": 1,
        "kt-num-gpu-experts": 4, "kt-method": "FP8",
        "kt-max-deferred-experts-per-token": 0,
        "kt-expert-placement-strategy": "uniform",
        "attention-backend": "flashinfer",
        "trust-remote-code": True, "mem-fraction-static": 0.85,
        "chunked-prefill-size": 4096, "max-running-requests": 1,
        "max-total-tokens": 32000,
        "enable-mixed-chunk": True, "tensor-parallel-size": 1,
        "disable-shared-experts-fusion": True,
    }))



    return C



# ============================================================================
# Hardware information collection
# ============================================================================

def collect_hardware_info():
    """Collect hardware information for the benchmark report."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {},
        "cpu": {},
        "memory": {},
        "isa_features": {},
    }

    uname = platform.uname()
    info["platform"] = {
        "system": uname.system,
        "node": uname.node,
        "release": uname.release,
        "machine": uname.machine,
    }

    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()

            for line in cpuinfo.split("\n"):
                if line.startswith("model name"):
                    info["cpu"]["model"] = line.split(":", 1)[1].strip()
                    break

            sockets = set()
            cores_per_socket = set()
            for line in cpuinfo.split("\n"):
                if "physical id" in line:
                    sockets.add(line.split(":", 1)[1].strip())
                if "cpu cores" in line:
                    cores_per_socket.add(line.split(":", 1)[1].strip())

            info["cpu"]["sockets"] = len(sockets) if sockets else 1
            info["cpu"]["cores_per_socket"] = int(list(cores_per_socket)[0]) if cores_per_socket else None
            info["cpu"]["logical_cores"] = os.cpu_count()

            flags = set()
            for line in cpuinfo.split("\n"):
                if line.startswith("flags"):
                    flags = set(line.split(":", 1)[1].strip().split())
                    break

            isa_checks = {
                "avx2": "avx2" in flags,
                "fma": "fma" in flags,
                "avx512f": "avx512f" in flags,
                "avx512bw": "avx512bw" in flags,
                "avx512_bf16": any(f in flags for f in ["avx512_bf16", "avx512bf16"]),
                "amx_tile": "amx_tile" in flags,
                "amx_int8": "amx_int8" in flags,
                "amx_bf16": "amx_bf16" in flags,
            }
            info["isa_features"] = isa_checks

            has_amx = all(isa_checks.get(f, False) for f in ["amx_tile", "amx_int8", "amx_bf16"])
            has_avx512 = isa_checks.get("avx512f", False)
            has_avx2 = isa_checks.get("avx2", False)

            if has_amx:
                info["cpu"]["tier"] = "AMX (Sapphire Rapids+)"
            elif has_avx512:
                info["cpu"]["tier"] = "AVX512"
            elif has_avx2:
                info["cpu"]["tier"] = "AVX2-only"
            else:
                info["cpu"]["tier"] = "Pre-AVX2"

        except Exception as e:
            info["cpu"]["error"] = str(e)

    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = float(line.split(":", 1)[1].split()[0])
                        info["memory"]["total_gb"] = round(mem_kb / (1024 * 1024), 2)
                    elif "MemAvailable" in line:
                        mem_kb = float(line.split(":", 1)[1].split()[0])
                        info["memory"]["available_gb"] = round(mem_kb / (1024 * 1024), 2)
        except Exception:
            pass

    # GPU info — collect both display strings and structured data
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            gpu_lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            info["gpu"] = gpu_lines
            vram_values = []
            for line in gpu_lines:
                parts = line.split(",")
                if len(parts) >= 2:
                    mem_str = parts[-1].strip()
                    mib = int("".join(c for c in mem_str if c.isdigit()))
                    vram_values.append(mib)
            if vram_values:
                info["gpu_structured"] = {
                    "count": len(vram_values),
                    "per_gpu_vram_mb": min(vram_values),
                    "vram_per_gpu": vram_values,
                }
    except Exception:
        pass

    return info


def print_hardware_info(hw_info):
    """Print hardware summary."""
    cpu = hw_info.get("cpu", {})
    mem = hw_info.get("memory", {})
    isa = hw_info.get("isa_features", {})
    gpus = hw_info.get("gpu", [])

    sockets = cpu.get("sockets", 1)
    cores_per_socket = cpu.get("cores_per_socket")
    physical_cores = (cores_per_socket * sockets) if cores_per_socket else "N/A"

    print("  Hardware Information:")
    print(f"    CPU:    {cpu.get('model', 'N/A')}")
    print(f"    Tier:   {cpu.get('tier', 'N/A')}")
    print(f"    Cores:  {cpu.get('logical_cores', 'N/A')} logical, "
          f"{physical_cores} physical, {sockets} socket(s)")
    print(f"    Memory: {mem.get('total_gb', 'N/A')} GB total, "
          f"{mem.get('available_gb', 'N/A')} GB available")
    isa_str = ", ".join(k for k, v in isa.items() if v)
    print(f"    ISA:    {isa_str if isa_str else 'N/A'}")
    if gpus:
        for i, g in enumerate(gpus):
            print(f"    GPU[{i}]: {g}")



# ============================================================================
# Server management
# ============================================================================

def find_appimage():
    """Locate the SGLang AppImage.  Priority: SGLANG_APPIMAGE env var >
    glob search in the script's directory."""
    env_path = os.environ.get("SGLANG_APPIMAGE")
    if env_path:
        p = os.path.abspath(env_path)
        if os.path.isfile(p):
            return p
        print(f"ERROR: SGLANG_APPIMAGE={env_path!r} does not exist")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    matches = sorted(glob.glob(os.path.join(script_dir, APPIMAGE_GLOB)))
    if matches:
        return matches[-1]

    print(f"ERROR: No {APPIMAGE_GLOB} found in {script_dir}")
    print("  Place an SGLang AppImage next to this script, or set SGLANG_APPIMAGE.")
    sys.exit(1)


def build_launch_cmd(args_dict, port, appimage_path):
    cmd = [appimage_path, "--host", "0.0.0.0", "--port", str(port)]
    for key, val in args_dict.items():
        if key.startswith("_"):
            continue
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(val)])
    return cmd


def get_env_vars(args_dict):
    env = os.environ.copy()
    for k, v in GLOBAL_ENV.items():
        env.setdefault(k, v)
    env.update(args_dict.get("_env", {}))
    return env


def tee_output(pipe, log_f, prefix="[server] "):
    try:
        for line in iter(pipe.readline, b''):
            decoded = line.decode("utf-8", errors="replace")
            log_f.write(decoded)
            log_f.flush()
            print(f"  {prefix}{decoded}", end="", flush=True)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def wait_for_server(port, timeout=SERVER_STARTUP_TIMEOUT):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline and not _skip_current_model:
        try:
            if requests.get(url, timeout=5).status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)
    return False


def print_runtime_server_config(port, log_path, log_file_handle=None):
    """Print resolved server config via HTTP and KT fields from server_args log line."""
    base = f"http://localhost:{port}"
    print("\n  --- Runtime server configuration (HTTP) ---")
    got_http = False
    for path in ("/server_info", "/get_server_info"):
        try:
            r = requests.get(f"{base}{path}", timeout=15)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            got_http = True
            try:
                print(json.dumps(r.json(), indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(r.text)
            break
        except requests.RequestException as exc:
            print(f"  ({path} failed: {exc})")
    if not got_http:
        print("  (no usable server_info response)")

    print("  --- KT-related fields from server log (server_args=...) ---")
    if log_file_handle is not None:
        try:
            log_file_handle.flush()
        except Exception:
            pass
    try:
        with open(log_path, "r", errors="replace") as lf:
            lines = lf.readlines()
    except OSError as exc:
        print(f"  (could not read log: {exc})")
        return
    found = None
    for line in reversed(lines):
        if "server_args=ServerArgs(" in line:
            found = line.strip()
            break
    if not found:
        print("  (no server_args=ServerArgs line in log yet)")
        return
    kt_matches = re.findall(r"kt_[a-z_0-9]+=[^,\)]+", found)
    if kt_matches:
        for m in sorted(set(kt_matches)):
            print(f"    {m}")
    else:
        print("  (no kt_* fields matched in server_args line)")
    print("  (full server_args line is in the model log file above)")


def kill_server(proc):
    if proc is None:
        return

    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, Exception):
        pass

    # Reap any zombie children left behind by the process group
    try:
        while True:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
    except ChildProcessError:
        pass


# ============================================================================
# Benchmark helpers
# ============================================================================

_FILLER_WORD = "word "  # ~1 token per word for most tokenizers

def generate_prompt(n_tokens):
    """Return a prompt string of approximately *n_tokens* tokens."""
    if n_tokens <= 2:
        return "hi"
    return "Repeat after me: " + _FILLER_WORD * n_tokens


def count_prompt_tokens_via_tokenize(port, served_name, prompt, timeout):
    """Return prompt token count from POST /tokenize (SGLang), or None if unavailable."""
    base = f"http://localhost:{port}"
    tmo = min(timeout, 120)
    candidates = [
        (f"{base}/tokenize", {"model": served_name, "text": prompt}),
        (f"{base}/tokenize", {"text": prompt}),
        (f"{base}/tokenize", {"prompt": prompt}),
        (f"{base}/v1/tokenize", {"model": served_name, "prompt": prompt}),
    ]

    def _len_from_obj(data):
        if data is None:
            return None
        if isinstance(data, list):
            return len(data) if data and isinstance(data[0], int) else None
        if not isinstance(data, dict):
            return None
        for key in ("tokens", "token_ids", "ids", "input_ids"):
            ids = data.get(key)
            if isinstance(ids, list) and ids and isinstance(ids[0], int):
                return len(ids)
        # OpenAI-style: {"data": [{"tokens": [...]}]}
        block = data.get("data")
        if isinstance(block, list) and block:
            return _len_from_obj(block[0])
        return None

    for url, body in candidates:
        try:
            r = requests.post(url, json=body, timeout=tmo)
            if r.status_code != 200:
                continue
            n = _len_from_obj(r.json())
            if n is not None:
                return n
        except (requests.RequestException, json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def run_single_benchmark(port, served_name, prompt, max_tokens, timeout):
    """Send a streaming /v1/completions request and measure TTFT / decode.

    Uses stream usage when available; otherwise /tokenize + SSE chunk fallback.
    Returns (prefill_tps, decode_tps, ttft, topt) or (None, None, None, None).
    """
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": served_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    first_token_time = None
    chunk_decode_count = 0
    usage_accum = {}
    t_start = time.perf_counter()

    try:
        with requests.post(url, json=payload, stream=True,
                           timeout=timeout) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                u = chunk.get("usage")
                if isinstance(u, dict):
                    for k, v in u.items():
                        if v is not None:
                            usage_accum[k] = v

                choices = chunk.get("choices") or []
                if choices:
                    text = choices[0].get("text", "")
                    if text and first_token_time is None:
                        first_token_time = time.perf_counter()
                    if text:
                        chunk_decode_count += 1
    except requests.exceptions.RequestException as exc:
        err = str(exc)
        if "CUDA" in err or "OOM" in err or "out of memory" in err.lower():
            return None, None, None, None
        print(f"    Request error: {err}")
        return None, None, None, None
    except Exception as exc:
        print(f"    Unexpected error: {exc}")
        return None, None, None, None

    t_end = time.perf_counter()

    if first_token_time is None:
        return None, None, None, None

    ttft = first_token_time - t_start
    decode_time = t_end - first_token_time

    prompt_tokens = usage_accum.get("prompt_tokens")
    completion_tokens = usage_accum.get("completion_tokens")

    if prompt_tokens is None or prompt_tokens == 0:
        pt_fb = count_prompt_tokens_via_tokenize(
            port, served_name, prompt, timeout)
        if pt_fb is not None and pt_fb > 0:
            prompt_tokens = pt_fb
            print("    WARNING: prompt token count from /tokenize "
                  "(stream usage missing or zero prompt_tokens)")
        else:
            prompt_tokens = len(prompt.split())
            print("    WARNING: prompt token count from whitespace split "
                  "(no usage / /tokenize failed)")

    used_usage_decode = (
        completion_tokens is not None and completion_tokens > 0)
    if not used_usage_decode:
        completion_tokens = chunk_decode_count
        if completion_tokens == 0:
            return None, None, None, None
        print("    WARNING: decode token count from SSE chunks "
              "(stream usage missing or zero completion_tokens)")

    prefill_tps = prompt_tokens / ttft if ttft > 0 else None
    decode_tps = completion_tokens / decode_time if decode_time > 0 else None
    topt = ((decode_time / completion_tokens * 1000)
            if completion_tokens > 0 and decode_time > 0 else None)

    return prefill_tps, decode_tps, ttft, topt


# ============================================================================
# Results I/O
# ============================================================================

def save_results_csv(all_results, filepath, hw_info=None):
    """Write benchmark results to CSV incrementally (crash-safe)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        if hw_info:
            cpu = hw_info.get("cpu", {})
            mem = hw_info.get("memory", {})
            isa = hw_info.get("isa_features", {})
            gpus = hw_info.get("gpu", [])
            gpu_struct = hw_info.get("gpu_structured", {})

            writer.writerow([f"# Timestamp: {hw_info.get('timestamp', 'N/A')}"])
            writer.writerow([f"# CPU: {cpu.get('model', 'N/A')}"])
            writer.writerow([f"# CPU Tier: {cpu.get('tier', 'N/A')}"])
            sockets = cpu.get("sockets", 1)
            cps = cpu.get("cores_per_socket", "N/A")
            writer.writerow([f"# Cores: {cpu.get('logical_cores', 'N/A')} logical / "
                             f"{cps}x{sockets} physical"])
            writer.writerow([f"# Memory: {mem.get('total_gb', 'N/A')} GB total / "
                             f"{mem.get('available_gb', 'N/A')} GB available"])
            isa_str = " ".join(k for k, v in isa.items() if v)
            writer.writerow([f"# ISA: {isa_str if isa_str else 'N/A'}"])
            for i, g in enumerate(gpus):
                writer.writerow([f"# GPU[{i}]: {g}"])
            if gpu_struct:
                writer.writerow([f"# GPU Count: {gpu_struct.get('count', 'N/A')}"])
            writer.writerow(["#"])

        labels = [cfg[0] for cfg in BENCH_CONFIGS]
        repro_headers = [col for _, col in REPRO_KEYS]
        header = ["Model", "Metric"] + labels + repro_headers
        writer.writerow(header)

        for result in all_results:
            model = result["model"]
            model_args = result.get("args", {})
            repro_vals = [str(model_args.get(k, "")) for k, _ in REPRO_KEYS]

            output_lengths = [f"{cfg[2]}tokens" for cfg in BENCH_CONFIGS]
            writer.writerow([model, "Output length"] + output_lengths + repro_vals)

            prefills = []
            decodes = []
            ttfts = []
            topts = []
            for cfg in result["configs"]:
                p = cfg.get("prefill_tps")
                d = cfg.get("decode_tps")
                t = cfg.get("ttft")
                o = cfg.get("topt")
                prefills.append(f"{p:.1f}" if p is not None else "FAIL")
                decodes.append(f"{d:.1f}" if d is not None else "FAIL")
                ttfts.append(f"{t*1000:.0f}" if t is not None else "FAIL")
                topts.append(f"{o:.1f}" if o is not None else "FAIL")
            writer.writerow([model, "Prefill token/s"] + prefills + repro_vals)
            writer.writerow([model, "Decode  token/s"] + decodes + repro_vals)
            writer.writerow([model, "TTFT (ms)"] + ttfts + repro_vals)
            writer.writerow([model, "TOPT (ms/tok)"] + topts + repro_vals)


def save_results_json(all_results, filepath):
    """Persist results as JSON for cross-invocation merging."""
    try:
        with open(filepath, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
    except Exception as e:
        print(f"  WARNING: Could not save JSON results: {e}")


def print_summary_table(all_results):
    """Print the results in the standard ktransformers benchmark table format."""
    labels = [cfg[0] for cfg in BENCH_CONFIGS]
    max_tokens_list = [cfg[2] for cfg in BENCH_CONFIGS]

    for result in all_results:
        model = result["model"]
        configs = result["configs"]

        col_widths = []
        for i, label in enumerate(labels):
            mt = max_tokens_list[i]
            header_w = max(len(label), len(f"{mt}tokens"))
            p = configs[i].get("prefill_tps") if i < len(configs) else None
            d = configs[i].get("decode_tps") if i < len(configs) else None
            t = configs[i].get("ttft") if i < len(configs) else None
            o = configs[i].get("topt") if i < len(configs) else None
            ps = f"{p:.0f}" if p is not None else "FAIL"
            ds = f"{d:.1f}" if d is not None else "FAIL"
            ts = f"{t*1000:.0f}" if t is not None else "FAIL"
            os_ = f"{o:.1f}" if o is not None else "FAIL"
            data_w = max(len(ps), len(ds), len(ts), len(os_))
            col_widths.append(max(header_w, data_w) + 2)

        row_label_w = 17
        sep = "+-" + "-" * row_label_w + "-+" + "+".join(
            "-" * w for w in col_widths) + "+"

        def fmt_row(label, values):
            cells = []
            for v, w in zip(values, col_widths):
                cells.append(f"{v:^{w}}")
            return f"| {label:<{row_label_w}} |" + "|".join(cells) + "|"

        model_args = result.get("args", {})
        cfg_parts = [f"{col}={model_args.get(k, '?')}"
                     for k, col in REPRO_KEYS if k in model_args]
        cfg_line = "  | " + "  ".join(cfg_parts) if cfg_parts else ""

        print(f"\n  === {model} ===")
        if cfg_line:
            print(cfg_line)
        print(f"  {sep}")
        print(f"  {fmt_row('Prompt', labels)}")
        print(f"  {fmt_row('Output length', [f'{mt}tokens' for mt in max_tokens_list])}")
        print(f"  {sep}")

        prefill_vals = []
        decode_vals = []
        ttft_vals = []
        topt_vals = []
        for i in range(len(labels)):
            if i < len(configs):
                p = configs[i].get("prefill_tps")
                d = configs[i].get("decode_tps")
                t = configs[i].get("ttft")
                o = configs[i].get("topt")
            else:
                p, d, t, o = None, None, None, None
            prefill_vals.append(f"{p:.0f}" if p is not None else "FAIL")
            decode_vals.append(f"{d:.1f}" if d is not None else "FAIL")
            ttft_vals.append(f"{t*1000:.0f}" if t is not None else "FAIL")
            topt_vals.append(f"{o:.1f}" if o is not None else "FAIL")

        print(f"  {fmt_row('Prefill token/s', prefill_vals)}")
        print(f"  {fmt_row('Decode  token/s', decode_vals)}")
        print(f"  {fmt_row('TTFT (ms)', ttft_vals)}")
        print(f"  {fmt_row('TOPT (ms/tok)', topt_vals)}")
        print(f"  {sep}")


# ============================================================================
# Main
# ============================================================================

_skip_current_model = False
_last_sigint_time = 0.0
_DOUBLE_CTRL_C_INTERVAL = 2.0


def _sigint_handler(signum, frame):
    global _skip_current_model, _last_sigint_time
    now = time.time()
    if now - _last_sigint_time < _DOUBLE_CTRL_C_INTERVAL:
        print("\n  >>> Double Ctrl+C — aborting entire benchmark <<<")
        sys.exit(1)
    _last_sigint_time = now
    _skip_current_model = True
    print("\n  >>> Ctrl+C — skipping current model (press again within "
          f"{_DOUBLE_CTRL_C_INTERVAL:.0f}s to quit) <<<")


def parse_args():
    parser = argparse.ArgumentParser(
        description="ktransformers + sglang benchmark for all models")
    parser.add_argument(
        "--models-base", default=MODELS_BASE, metavar="DIR",
        help=f"Base directory containing model folders (default: {MODELS_BASE})")
    parser.add_argument(
        "models", nargs="*", metavar="MODEL",
        help="Model name(s) to benchmark (default: all)")
    return parser.parse_args()


def main():
    global _skip_current_model, MODELS_BASE

    cli = parse_args()
    MODELS_BASE = cli.models_base

    print("=" * 72)
    print("  ktransformers + sglang benchmark")
    print("=" * 72)

    print(f"\n  Models base: {MODELS_BASE}")

    # --- AppImage ---
    appimage = find_appimage()
    print(f"  AppImage: {appimage}")

    # --- Hardware ---
    hw_info = collect_hardware_info()
    print()
    print_hardware_info(hw_info)

    # --- Model configs ---
    configs = get_model_configs()

    # --- Filter by CLI args ---
    if cli.models:
        filtered = [(n, s, a) for n, s, a in configs if n in cli.models]
        unknown = set(cli.models) - {n for n, _, _ in configs}
        if unknown:
            print(f"\n  WARNING: unknown model(s): {', '.join(sorted(unknown))}")
        configs = filtered
        print(f"\n  Selected {len(configs)} model(s): "
              + ", ".join(n for n, _, _ in configs))
    else:
        print(f"\n  Running all {len(configs)} models")

    if not configs:
        print("  No models to benchmark. Exiting.")
        return

    # --- Prepare results dir ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, "perf.csv")
    json_path = os.path.join(RESULTS_DIR, "perf_data.json")
    print(f"  Results will be saved to: {csv_path}")

    # --- Load previous results from JSON sidecar ---
    current_models = {n for n, _, _ in configs}
    all_results = []
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                previous = json.load(f)
            all_results = [r for r in previous if r["model"] not in current_models]
            if all_results:
                print(f"  Loaded {len(all_results)} previous model result(s) from {json_path}")
        except Exception as e:
            print(f"  WARNING: Could not load previous results: {e}")

    # --- Benchmark loop ---
    original_handler = signal.getsignal(signal.SIGINT)

    for idx, (model_name, served_name, args) in enumerate(configs):
        _skip_current_model = False
        port = BASE_PORT + idx
        log_path = os.path.join(RESULTS_DIR, f"{model_name}_{timestamp}.log")

        banner = f"  [{idx+1}/{len(configs)}] {model_name}  "
        print(f"\n\n{'#'*72}")
        print(f"{'#'*72}")
        print(f"##{'':^68}##")
        print(f"##{banner:^68}##")
        print(f"##{'':^68}##")
        print(f"{'#'*72}")
        print(f"{'#'*72}")

        cmd = build_launch_cmd(args, port, appimage)
        env = get_env_vars(args)
        print(f"  Port: {port}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Log: {log_path}")

        proc = None
        result = {"model": model_name, "args": args, "configs": []}

        try:
            log_f = open(log_path, "w")
            proc = subprocess.Popen(
                cmd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )

            tee_thread = threading.Thread(
                target=tee_output, args=(proc.stdout, log_f), daemon=True)
            tee_thread.start()

            signal.signal(signal.SIGINT, _sigint_handler)

            print(f"\n  Waiting for server (timeout {SERVER_STARTUP_TIMEOUT}s)...")
            if not wait_for_server(port):
                if _skip_current_model:
                    print("  Skipped by user during server startup.")
                else:
                    print("  ERROR: Server failed to start. Skipping model.")
                result["configs"] = [{"label": c[0], "prefill_tps": None,
                                      "decode_tps": None, "ttft": None,
                                      "topt": None} for c in BENCH_CONFIGS]
                all_results.append(result)
                save_results_csv(all_results, csv_path, hw_info)
                save_results_json(all_results, json_path)
                continue

            if _skip_current_model:
                print("  Skipped by user during server startup.")
                result["configs"] = [{"label": c[0], "prefill_tps": None,
                                      "decode_tps": None, "ttft": None,
                                      "topt": None} for c in BENCH_CONFIGS]
                all_results.append(result)
                save_results_csv(all_results, csv_path, hw_info)
                save_results_json(all_results, json_path)
                continue

            print("  Server is ready.")
            try:
                log_f.flush()
            except Exception:
                pass
            print_runtime_server_config(port, log_path, log_f)

            # Warmup
            print("  Warming up...")
            try:
                wp, wd, _, _ = run_single_benchmark(
                    port, served_name, "hi", 5, WARMUP_TIMEOUT)
                if wp is not None:
                    print(f"  Warmup OK (prefill={wp:.0f}, decode={wd:.1f})")
                else:
                    print("  Warmup returned no data (may be fine)")
            except Exception as e:
                print(f"  Warmup error (continuing): {e}")

            if _skip_current_model:
                print("  Skipped by user.")
                result["configs"] = [{"label": c[0], "prefill_tps": None,
                                      "decode_tps": None, "ttft": None,
                                      "topt": None} for c in BENCH_CONFIGS]
                all_results.append(result)
                save_results_csv(all_results, csv_path, hw_info)
                save_results_json(all_results, json_path)
                continue

            # Benchmark each config
            for cfg_label, approx_tokens, max_gen in BENCH_CONFIGS:
                if _skip_current_model:
                    result["configs"].append({"label": cfg_label,
                                              "prefill_tps": None,
                                              "decode_tps": None,
                                              "ttft": None, "topt": None})
                    continue

                prompt = generate_prompt(approx_tokens)
                print(f"\n  Bench: {cfg_label} "
                      f"(~{approx_tokens} tok prompt, {max_gen} tok gen)...")

                try:
                    p_tps, d_tps, ttft, topt = run_single_benchmark(
                        port, served_name, prompt, max_gen, REQUEST_TIMEOUT)
                except Exception as e:
                    print(f"    Error: {e}")
                    p_tps, d_tps, ttft, topt = None, None, None, None

                if p_tps is not None:
                    print(f"    Prefill: {p_tps:.1f} tok/s, "
                          f"Decode: {d_tps:.1f} tok/s, "
                          f"TTFT: {ttft*1000:.0f} ms, "
                          f"TOPT: {topt:.1f} ms/tok")
                else:
                    print(f"    FAIL")

                result["configs"].append({
                    "label": cfg_label,
                    "prefill_tps": p_tps,
                    "decode_tps": d_tps,
                    "ttft": ttft,
                    "topt": topt,
                })

        except Exception:
            print(f"  Exception during benchmark:")
            traceback.print_exc()
            if len(result["configs"]) < len(BENCH_CONFIGS):
                for c in BENCH_CONFIGS[len(result["configs"]):]:
                    result["configs"].append({"label": c[0],
                                              "prefill_tps": None,
                                              "decode_tps": None,
                                              "ttft": None, "topt": None})
        finally:
            signal.signal(signal.SIGINT, original_handler)
            print(f"\n  Stopping server for {model_name}...")
            kill_server(proc)
            try:
                log_f.close()
            except Exception:
                pass

        all_results.append(result)
        save_results_csv(all_results, csv_path, hw_info)
        save_results_json(all_results, json_path)
        print(f"  Results saved ({len(all_results)}/{len(configs)} models done)")

        if idx < len(configs) - 1:
            print(f"  Waiting {INTER_MODEL_WAIT}s before next model...")
            time.sleep(INTER_MODEL_WAIT)

    # --- Summary ---
    print(f"\n{'='*72}")
    print("  BENCHMARK COMPLETE")
    print(f"{'='*72}")
    print(f"\n  CSV: {csv_path}")
    print_summary_table(all_results)


if __name__ == "__main__":
    main()
PYTHON_SCRIPT_EOF
ok "benchmark_all_models.py generated"

# ============================================================================
# Step 5: Download, benchmark, and delete each model one by one
# ============================================================================

export APPIMAGE_EXTRACT_AND_RUN=1
TOTAL=${#MODEL_LIST[@]}
IDX=0

for entry in "${MODEL_LIST[@]}"; do
    IFS='|' read -r M_NAME M_REPO <<< "$entry"
    IDX=$((IDX + 1))

    log "[$IDX/$TOTAL] Processing model: $M_NAME"

    # --- Download ---
    log "Downloading: $M_REPO -> $MODELS_BASE/$M_NAME"
    huggingface-cli download "$M_REPO" \
        --local-dir "$MODELS_BASE/$M_NAME" \
        --local-dir-use-symlinks False

    # --- Benchmark ---
    log "Benchmarking: $M_NAME"
    echo "    AppImage:  $SCRIPT_DIR/$APPIMAGE_NAME"
    echo "    Model:     $MODELS_BASE/$M_NAME"
    echo "    Results:   $SCRIPT_DIR/perf/perf.csv"
    echo ""
    python3 "$SCRIPT_DIR/benchmark_all_models.py" \
        --models-base "$MODELS_BASE" \
        "$M_NAME" || true

    # --- Delete ---
    log "Deleting model: $MODELS_BASE/$M_NAME"
    rm -rf "$MODELS_BASE/$M_NAME"

    ok "[$IDX/$TOTAL] $M_NAME done"
done

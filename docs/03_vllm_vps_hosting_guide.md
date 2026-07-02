# Hosting a Qwen Model on a VPS Using vLLM — Production Guide

This guide deploys the fine-tuned model from Task 1 (merged safetensors at
`outputs/qwen3-8b-vedaz-merged-fp16/`, or any Qwen3/Qwen2.5 checkpoint — the steps are
identical) as an **OpenAI-API-compatible** HTTP service on a bare Ubuntu VPS, fronted by Nginx +
TLS, running as a supervised systemd service. Every step includes the exact terminal command.
Companion config files referenced below live in [`deployment/`](../deployment/).

---

## 1. Server requirements

| Component | Minimum | Recommended (this project's 8B model) |
|---|---|---|
| GPU | 1× NVIDIA GPU, 16GB VRAM (Qwen3-8B in fp16 needs ~16GB just for weights + KV cache headroom) | 1× NVIDIA L4 / A10G / RTX 4090 (24GB), or 1× A100 40GB for larger batch/concurrency |
| vCPUs | 4 | 8+ |
| System RAM | 16 GB | 32 GB+ |
| Disk | 60 GB SSD (model weights ~16GB fp16 + OS + logs) | 150 GB+ NVMe SSD (room for multiple model versions, quant variants) |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |
| Network | Public IPv4, port 443 reachable | Same, plus a DNS A record pointing at the box |
| Providers that fit this spec | Lambda Cloud, RunPod, Vast.ai, Paperspace, AWS `g5.xlarge`/`g6.xlarge`, GCP `g2-standard-8`, Azure `NC A10 v4` | Same providers, larger tier |

Confirm the GPU is visible to the OS before doing anything else:

```bash
lspci | grep -i nvidia
```

---

## 2. Ubuntu base setup

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential curl wget git ca-certificates gnupg lsb-release \
    software-properties-common ufw htop tmux unzip
sudo timedatectl set-timezone UTC
sudo reboot
```

Create a dedicated non-root service user (never run the model server as root):

```bash
sudo adduser --disabled-password --gecos "" vllm
sudo usermod -aG sudo vllm
su - vllm
```

---

## 3. GPU driver installation

```bash
# Remove any conflicting/old driver first
sudo apt purge -y '*nvidia*' 2>/dev/null || true

# Add the official NVIDIA CUDA apt repo (also provides the driver package)
distro="ubuntu2204"
wget https://developer.download.nvidia.com/compute/cuda/repos/${distro}/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install the recommended production driver (open kernel modules, matches CUDA 12.4)
sudo apt install -y nvidia-driver-550

sudo reboot
```

Verify after reboot:

```bash
nvidia-smi
```

Expected output includes the GPU name, driver version (≥550.x), and CUDA version (≥12.4)
in the top-right corner of the table.

---

## 4. CUDA toolkit installation

vLLM ships CUDA-bundled wheels for the common versions, so a full toolkit install is only
required if you plan to build custom kernels (e.g. `flash-attn` from source). Install it anyway
— it's needed for `nvcc`-based diagnostics and optional flash-attn builds:

```bash
sudo apt install -y cuda-toolkit-12-4

echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version   # confirm 12.4
```

---

## 5. Python and virtual environment

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

mkdir -p ~/qwen-serving && cd ~/qwen-serving
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

Every command from here on assumes `~/qwen-serving/.venv` is active
(prompt shows `(.venv)`).

---

## 6. vLLM installation

```bash
# vLLM's published wheel bundles a matching torch + CUDA runtime — do not pre-install a
# different torch version, it will conflict.
pip install vllm==0.7.1

# Sanity check
python -c "import vllm; print(vllm.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` returns `False`, stop here and re-check the driver install
(§3) before continuing — nothing downstream will work.

---

## 7. Hugging Face authentication

Needed to pull the base Qwen weights (if serving the base model) or to push/pull your
fine-tuned model from a private HF repo. Skip this step if you are copying the merged model
directory over `scp` instead (§8, option B).

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# Paste your token (https://huggingface.co/settings/tokens) when prompted.
# Use a READ-scoped token on the server — never store a write token on a public-facing box.
```

Non-interactive alternative (for automated provisioning scripts):

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
```

---

## 8. Downloading / transferring the model

**Option A — pull from Hugging Face Hub** (base Qwen3-8B, or your fine-tune if pushed there):

```bash
mkdir -p ~/models
huggingface-cli download Qwen/Qwen3-8B --local-dir ~/models/qwen3-8b --local-dir-use-symlinks False
```

**Option B — transfer your locally fine-tuned merged model** (the output of
`merge_and_export.py` in this repo) from your training machine to the VPS:

```bash
# Run from your TRAINING machine, not the VPS:
rsync -avzP --progress \
  ./outputs/qwen3-8b-vedaz-merged-fp16/ \
  vllm@<VPS_PUBLIC_IP>:~/models/qwen3-8b-vedaz/
```

Verify the directory has the expected files on the VPS:

```bash
ls -lh ~/models/qwen3-8b-vedaz/
# expect: config.json, generation_config.json, model-*.safetensors, tokenizer.json,
#         tokenizer_config.json, special_tokens_map.json
```

---

## 9. Serving the model (manual smoke test)

Before wiring up systemd, run it in the foreground once to confirm it boots cleanly:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen3-8b-vedaz \
  --served-model-name qwen3-8b-vedaz \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --api-key "$(openssl rand -hex 32)"
```

Watch the log for `Uvicorn running on http://0.0.0.0:8000` — that means the server is ready.
Note the `--api-key` value it printed (or generate/set your own) — you will need it for every
request. Press `Ctrl+C` to stop once confirmed; §14 wires this into systemd so it survives
reboots and crashes.

---

## 10. The OpenAI-compatible API

vLLM's `api_server` implements the OpenAI REST surface, so any OpenAI SDK/tool works against it
by just changing the base URL. Key endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat-formatted generation (what this project uses) |
| `POST /v1/completions` | Raw text completion |
| `GET /v1/models` | List served model name(s) |
| `GET /health` | Liveness probe (no auth required) — used by systemd/monitoring |
| `GET /metrics` | Prometheus metrics (§17) |

### Authentication

Every `/v1/*` call requires `Authorization: Bearer <API_KEY>` where `<API_KEY>` is whatever you
passed to `--api-key` at startup. `/health` and `/metrics` are intentionally unauthenticated so
load balancers/monitoring can probe them without a secret — this is why §16 (firewall) and §15
(Nginx) must ensure those aren't exposed to the raw internet unnecessarily beyond the LB.

### curl examples

```bash
export API_KEY="paste-the-key-from-step-9"
export BASE_URL="http://localhost:8000/v1"

# List models
curl -s $BASE_URL/models -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# Non-streaming chat completion
curl -s $BASE_URL/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-vedaz",
    "messages": [
      {"role": "system", "content": "You are Vedaz'"'"'s AI Vedic astrologer. Give balanced, compassionate, non-fatalistic guidance."},
      {"role": "user", "content": "Mera career kab tak stabilize hoga?"}
    ],
    "temperature": 0.7,
    "max_tokens": 400
  }' | python3 -m json.tool

# Health check (no auth)
curl -s http://localhost:8000/health -o /dev/null -w "HTTP %{http_code}\n"
```

### Streaming response (curl, SSE)

```bash
curl -N $BASE_URL/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-vedaz",
    "messages": [{"role": "user", "content": "Mujhe business mein loss ho raha hai, kya karu?"}],
    "stream": true,
    "max_tokens": 400
  }'
```

Each line arrives as `data: {"choices":[{"delta":{"content":"..."}}], ...}` followed by a final
`data: [DONE]`.

### Python examples

```python
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="https://your-domain.com/v1", api_key="paste-the-key-from-step-9")

# Non-streaming
resp = client.chat.completions.create(
    model="qwen3-8b-vedaz",
    messages=[
        {"role": "system", "content": "You are Vedaz's AI Vedic astrologer. Give balanced, compassionate, non-fatalistic guidance."},
        {"role": "user", "content": "What does my Saturn transit mean for my career this year?"},
    ],
    temperature=0.7,
    max_tokens=400,
)
print(resp.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="qwen3-8b-vedaz",
    messages=[{"role": "user", "content": "मेरी शादी में देरी क्यों हो रही है?"}],
    stream=True,
    max_tokens=400,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Health checks

```bash
# Simple liveness
curl -f http://localhost:8000/health || echo "SERVER DOWN"

# Scriptable readiness probe used by systemd/monitoring (see deployment/scripts/healthcheck.sh)
bash deployment/scripts/healthcheck.sh
```

---

## 11. Systemd service (production process supervision)

Use the unit file at [`deployment/systemd/qwen-vllm.service`](../deployment/systemd/qwen-vllm.service):

```ini
[Unit]
Description=vLLM OpenAI-compatible server (qwen3-8b-vedaz)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=vllm
Group=vllm
WorkingDirectory=/home/vllm/qwen-serving
EnvironmentFile=/home/vllm/qwen-serving/.env
ExecStart=/home/vllm/qwen-serving/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /home/vllm/models/qwen3-8b-vedaz \
    --served-model-name qwen3-8b-vedaz \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --api-key ${VLLM_API_KEY}
Restart=always
RestartSec=5
LimitNOFILE=65536
StandardOutput=append:/var/log/vllm/vllm.log
StandardError=append:/var/log/vllm/vllm-error.log

[Install]
WantedBy=multi-user.target
```

Note it binds to `127.0.0.1` (not `0.0.0.0`) — the model server is never exposed directly to
the internet; Nginx (§12) is the only public-facing process. This is a deliberate defense-in-
depth choice, not an oversight.

Install and start it:

```bash
sudo mkdir -p /var/log/vllm && sudo chown vllm:vllm /var/log/vllm

echo "VLLM_API_KEY=$(openssl rand -hex 32)" > ~/qwen-serving/.env
cat ~/qwen-serving/.env   # save this value — you'll put it in Nginx/client configs too

sudo cp deployment/systemd/qwen-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qwen-vllm
sudo systemctl start qwen-vllm

# Check status / logs
sudo systemctl status qwen-vllm
sudo journalctl -u qwen-vllm -f
tail -f /var/log/vllm/vllm.log
```

---

## 12. Reverse proxy with Nginx

Config at [`deployment/nginx/qwen.conf`](../deployment/nginx/qwen.conf):

```nginx
upstream vllm_backend {
    server 127.0.0.1:8000;
    keepalive 64;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Streaming responses (SSE) must not be buffered
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        # Generation can legitimately take tens of seconds for long replies
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        # Basic rate limiting (see limit_req_zone below) to blunt abuse before it reaches vLLM
        limit_req zone=api_limit burst=20 nodelay;
    }

    location /health {
        proxy_pass http://vllm_backend/health;
        access_log off;
    }
}

limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
```

Install and enable:

```bash
sudo apt install -y nginx
sudo cp deployment/nginx/qwen.conf /etc/nginx/sites-available/qwen.conf
sudo ln -s /etc/nginx/sites-available/qwen.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

---

## 13. SSL with Let's Encrypt

```bash
sudo apt install -y certbot python3-certbot-nginx

# Point your domain's DNS A record at the VPS IP BEFORE running this
sudo certbot --nginx -d your-domain.com --redirect --agree-tos -m you@example.com --no-eff-email

# Certbot edits the Nginx config in place to add the 443 server block + auto-redirect from 80.
# Verify auto-renewal is registered:
sudo systemctl status certbot.timer
sudo certbot renew --dry-run
```

After this, all client examples in §10 should use `https://your-domain.com/v1` instead of
`http://localhost:8000/v1`.

---

## 14. Firewall

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow OpenSSH        # or your custom SSH port — check before enabling!
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status verbose
```

Port 8000 (vLLM) is deliberately **not** opened — it's only reachable via `127.0.0.1` from
Nginx on the same host (§11). If running multi-node (§19), use a private VPC network / security
group rule scoped to the internal subnet instead of exposing 8000 publicly.

---

## 15. Monitoring

vLLM exposes Prometheus metrics natively at `/metrics` (queue depth, running/waiting requests,
GPU KV-cache usage, time-per-output-token, time-to-first-token).

```bash
# Quick manual check
curl -s http://localhost:8000/metrics | grep vllm | head -20
```

Minimal Prometheus + Grafana stack (optional but recommended for anything beyond a demo):

```bash
# Prometheus (scrapes vLLM's /metrics every 15s)
sudo useradd --no-create-home --shell /usr/sbin/nologin prometheus
wget https://github.com/prometheus/prometheus/releases/latest/download/prometheus-linux-amd64.tar.gz
tar xzf prometheus-linux-amd64.tar.gz && cd prometheus-*/
sudo cp prometheus promtool /usr/local/bin/
sudo mkdir -p /etc/prometheus
cat <<'EOF' | sudo tee /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['127.0.0.1:8000']
EOF
sudo prometheus --config.file=/etc/prometheus/prometheus.yml &

# GPU-level metrics (utilization, VRAM, temperature, power)
sudo apt install -y nvidia-utils-550
watch -n 2 nvidia-smi
# or export as Prometheus metrics via dcgm-exporter for a proper dashboard
```

Key metrics to alert on: `vllm:num_requests_waiting` (queue building up → scale out or
increase `--max-num-seqs`), `vllm:gpu_cache_usage_perc` (approaching 100% → OOM risk, see §18),
`vllm:time_to_first_token_seconds` (p99 latency SLO).

---

## 16. Logging

```bash
# Application logs (from the systemd unit's StandardOutput/StandardError, §11)
tail -f /var/log/vllm/vllm.log
tail -f /var/log/vllm/vllm-error.log

# Journald (systemd's own capture, redundant with the above but useful for `since`/`until` queries)
sudo journalctl -u qwen-vllm --since "1 hour ago"

# Nginx access/error logs (per-request audit trail, request latency, status codes)
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

Rotate logs so a long-running server doesn't fill the disk:

```bash
cat <<'EOF' | sudo tee /etc/logrotate.d/vllm
/var/log/vllm/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF
```

---

## 17. Performance tuning

| Flag | Effect | Recommendation for an 8B model on a 24GB GPU |
|---|---|---|
| `--gpu-memory-utilization` | Fraction of VRAM vLLM pre-allocates for weights + KV cache | `0.90` — leave ~10% headroom for CUDA context/fragmentation |
| `--max-model-len` | Max context length served | `4096` (matches training config; raising it shrinks available KV-cache slots for concurrency) |
| `--max-num-seqs` | Max concurrent sequences batched together | Start at `256`, tune down if you see OOM under load, up if GPU is under-utilized with low concurrency |
| `--enable-prefix-caching` | Reuses KV cache across requests sharing a prefix (e.g. the same system prompt) | **Enable** — this workload's system prompt is long (up to 371 tokens) and shared across nearly every request, so prefix caching gives a large real-world win |
| `--dtype bfloat16` | Compute precision | Keep `bfloat16` — matches training precision, avoids fp16 overflow edge cases |
| `--quantization` | Serve a quantized checkpoint (AWQ/GPTQ/FP8) instead of fp16 | Use if VRAM-constrained or want higher throughput; see §18 |
| `--swap-space` | CPU RAM used as overflow for KV cache under memory pressure | `4` (GB) is a safe default; raises resilience to bursty concurrency at a latency cost |

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen3-8b-vedaz \
  --served-model-name qwen3-8b-vedaz \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --swap-space 4 \
  --api-key "$VLLM_API_KEY"
```

### Tensor parallelism (multi-GPU, single node)

If you have more than one GPU on the box, split the model across them:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/qwen3-8b-vedaz \
  --tensor-parallel-size 2 \
  ... # other flags as above
```

`--tensor-parallel-size` must evenly divide the number of attention heads (Qwen3-8B has 32
query heads, so valid values are 1, 2, 4, 8, 16, 32 — in practice 2 or 4 for this model size).
Tensor parallelism trades cross-GPU NVLink/PCIe communication overhead for lower per-GPU memory
pressure and higher aggregate throughput — worthwhile once a single GPU is throughput-bound, not
necessary for an 8B model's memory footprint alone on a 24GB+ card.

### Memory optimization

- **Quantize the served weights** if VRAM is the bottleneck rather than compute: export an
  AWQ or GPTQ 4-bit version of the merged model (`pip install autoawq`, then
  `--quantization awq`) — roughly halves weight VRAM vs bf16 with a small quality cost, freeing
  that memory for a larger KV cache (more concurrent users).
- **Lower `--max-model-len`** if you don't need the full 4096 — KV cache size scales linearly
  with max context, so serving at 2048 roughly doubles the number of concurrent sequences you
  can fit.
- **`--enable-prefix-caching`** (above) is effectively free memory-efficiency for this
  workload's shared, long system prompt.
- Watch `vllm:gpu_cache_usage_perc` in `/metrics` (§15) — sustained >95% means you're one
  traffic spike away from request rejection; reduce `--max-num-seqs` or add a GPU.

---

## 18. Common errors and debugging

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` on startup | `--gpu-memory-utilization` too high for actual free VRAM, or another process holds GPU memory | `nvidia-smi` to check for stray processes; lower `--gpu-memory-utilization` to `0.80`; kill zombie python processes with `sudo fuser -k /dev/nvidia0` |
| `torch.cuda.is_available() == False` | Driver/CUDA mismatch, or GPU not passed through (VM/container) | Re-run `nvidia-smi`; confirm `nvidia-container-toolkit` is installed if running vLLM inside Docker; reboot after driver install |
| Server starts but every request 401s | Missing/incorrect `Authorization: Bearer` header, or `.env`'s `VLLM_API_KEY` doesn't match what the client sends | `echo $VLLM_API_KEY` on the server, compare byte-for-byte with the client's key |
| Requests hang / never return with `stream: true` | Nginx buffering SSE (missing `proxy_buffering off;`) | Confirm §12's Nginx config has `proxy_buffering off` and `chunked_transfer_encoding on` |
| `502 Bad Gateway` from Nginx | vLLM process crashed or not yet finished loading weights (can take 30–90s for an 8B model) | `sudo systemctl status qwen-vllm`; check `/var/log/vllm/vllm-error.log`; wait for `Uvicorn running` log line before hitting Nginx |
| Slow first request, fast afterward | Expected — CUDA graph capture / kernel warmup on first inference | Not a bug; optionally send a throwaway warmup request right after startup in your deploy script |
| `ImportError: flash_attn` at startup | Optional flash-attn wheel not installed or CUDA ABI mismatch | vLLM works fine without it (falls back to its own attention kernels) — only install flash-attn if you've confirmed a measured throughput need |
| Model output looks like base Qwen, not your fine-tune | Pointed `--model` at the base checkpoint instead of the merged fine-tuned directory | Double check `~/models/qwen3-8b-vedaz` is the `merge_and_export.py` output, not the raw HF download |
| High latency under concurrent load | `--max-num-seqs` too low for traffic, or KV cache exhausted causing request queuing | Check `vllm:num_requests_waiting` in `/metrics`; raise `--max-num-seqs` if VRAM allows, else scale out (§19) |

General debugging flow:

```bash
sudo systemctl status qwen-vllm        # is the process even running?
sudo journalctl -u qwen-vllm -n 200    # last 200 log lines
nvidia-smi                             # GPU memory/utilization snapshot
curl -v http://localhost:8000/health   # is the API layer responding at all?
curl -s http://localhost:8000/metrics | grep -E "waiting|cache_usage"  # is it overloaded?
```

---

## 19. Scaling and multi-GPU deployment

Three scaling paths, in order of increasing complexity:

1. **Vertical (single node, more GPUs, tensor parallelism)** — §17's
   `--tensor-parallel-size`. Simplest; good up to 4–8 GPUs on one box.
2. **Horizontal (multiple independent replicas behind a load balancer)** — run the same
   systemd service on N VPS instances, each fully independent, and put them behind Nginx
   `upstream` with multiple `server` lines (round-robin) or a managed load balancer
   (AWS ALB / GCP HTTP LB):

   ```nginx
   upstream vllm_backend {
       least_conn;
       server 10.0.0.11:8000;
       server 10.0.0.12:8000;
       server 10.0.0.13:8000;
       keepalive 64;
   }
   ```

   This is the right default for scaling a stateless chat API — vLLM requests are independent,
   so horizontal replicas need no coordination.
3. **Distributed (pipeline + tensor parallelism across multiple nodes)** — only relevant for
   models too large for one node's aggregate VRAM (e.g. Qwen3-235B-A22B MoE); not applicable to
   an 8B model. Uses `--pipeline-parallel-size` with Ray; see vLLM's distributed serving docs if
   you later fine-tune a larger Qwen3 variant.

For this project's 8B model, **horizontal replicas behind a load balancer** is the recommended
production scaling path — it's operationally simpler than TP/PP, isolates blast radius (one
instance crashing doesn't take down the whole service), and scales cost linearly and
predictably with traffic.

---

## 20. Production best practices

- **Never bind vLLM to `0.0.0.0` on a public interface.** Bind `127.0.0.1`, front with Nginx +
  TLS (§11–13).
- **Always set `--api-key`.** An unauthenticated LLM endpoint on the open internet will be found
  and abused for free inference within hours.
- **Pin exact package versions** (`requirements.txt`) — vLLM, torch, and CUDA driver versions
  are tightly coupled; an unplanned `pip install -U vllm` on a production box can silently break
  the CUDA ABI.
  Test upgrades in staging first.
- **Warm up the model after every deploy** (send a handful of representative requests) before
  routing real traffic to it — first-request CUDA graph capture latency (§18) would otherwise
  hit a real user.
- **Separate the model-serving user from your login user** (§2's `vllm` service account) —
  limits blast radius if the API layer is ever compromised.
- **Keep the merged model directory read-only** for the service account after deployment
  (`chmod -R a-w`) to reduce tamper risk.
- **Version your served model name** (`--served-model-name qwen3-8b-vedaz-v2`) on each
  fine-tune iteration so clients can pin to a known-good version during rollout and you can run
  A/B or canary traffic between versions on two ports/upstreams.
- **Automate the whole install** (§2–14) as a cloud-init / Ansible playbook once you've
  validated it manually once — manual VPS setup does not survive a second server reliably.

## 21. Cost optimization

- **Right-size the GPU.** An 8B model does not need an A100/H100 for low-to-moderate traffic —
  an L4/A10G (24GB) is 3–5x cheaper per hour and is VRAM-sufficient per §1/§17.
- **Quantize to raise effective throughput per dollar** — AWQ/GPTQ 4-bit (§17) lets one GPU
  serve more concurrent users, which is usually a better ROI than adding a second GPU.
- **Use spot/preemptible instances for batch evaluation workloads** (Task 5's scripts) — only
  the always-on production chat endpoint needs an on-demand/reserved instance.
- **Enable prefix caching** (§17) — this workload's long, largely-shared system prompt means
  prefix caching directly reduces wasted compute per request, which is realized as either lower
  latency or more concurrent users per GPU-hour.
- **Scale replicas to traffic, not to peak** — horizontal scaling (§19) means you can run a
  single small instance off-peak and add replicas only during measured demand, via a simple
  cron-based or metrics-based autoscaler watching `vllm:num_requests_waiting`.
- **Monitor idle GPU time.** If `nvidia-smi` shows sustained <20% utilization, the instance is
  oversized for current traffic — downsize before adding more infrastructure.

## 22. Security recommendations

- API key auth (§10) on every `/v1/*` route; rotate the key periodically and on any suspected
  leak (`openssl rand -hex 32` to generate a new one, update `.env`, `systemctl restart qwen-vllm`).
- TLS everywhere (§13) — never send the API key over plain HTTP.
- Firewall default-deny (§14); only 22 (SSH, ideally key-only + non-default port), 80, 443 open.
- Rate limiting at the Nginx layer (§12's `limit_req_zone`) to blunt both abuse and accidental
  client-side retry storms before they reach the GPU process.
- Disable SSH password auth, key-only:
  ```bash
  sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
  sudo systemctl restart sshd
  ```
- Keep the OS patched: `sudo apt update && sudo apt upgrade -y` on a schedule (unattended-upgrades
  recommended for security patches: `sudo apt install -y unattended-upgrades`).
- Don't log full request/response bodies in production (§16) if user prompts may contain PII
  (birth date/time/place, in this project's case) — log metadata (latency, token counts, status
  codes) and keep full-content logging opt-in/short-retention and access-controlled, consistent
  with basic data-privacy hygiene for a service that collects birth details.
- Review `/metrics` and `/health` exposure — they're unauthenticated by design (§10); make sure
  Nginx/firewall don't inadvertently expose them beyond your monitoring network if that matters
  for your threat model.

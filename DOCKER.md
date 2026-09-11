# Step 20: Docker

Run these commands in PowerShell from `D:\projects\arabic-tool-llm`.

## Verified on 2026-09-11

- Runtime image and optional test image built successfully.
- Native Windows tests: 9 passed, 2 library deprecation warnings.
- Linux container tests: 9 passed, the same 2 warnings.
- Dependency check: no broken requirements.
- GPU check: PyTorch 2.9.1+cu126, CUDA 12.6, Quadro P1000, compute capability 6.1; a CUDA tensor operation succeeded.
- Model startup: 196 NF4 4-bit linear layers, V2 LoRA loaded, zero trainable parameters during inference.
- Real HTTP `/health`: 200, `healthy`, `model_loaded: true`.
- Real Arabic `/chat`: `فين الطلب رقم 3147 وصل؟` returned `get_order_status`, order 3147, `out_for_delivery`. This individual request took 5.65 seconds; this is a smoke test, not a latency benchmark.
- Whitespace-only `/chat`: 422. A supplied `X-Request-ID` was returned unchanged.
- Container `arabic-tool-llm` was left running and healthy on `http://localhost:8000`.

The container already exists on this machine. Use `docker start arabic-tool-llm` after stopping it; remove the old container before repeating the full `docker run` command below.

## Requirements

- Docker Desktop running Linux containers with its WSL 2 backend.
- NVIDIA GPU available to Docker (`--gpus all`) and a compatible Windows driver.
- The trained adapter files must exist in `models\qwen3-0.6b-tool-calling-v2-lora`, including `adapter_config.json` and `adapter_model.safetensors`.
- Internet access for the first image build and first Qwen download. CUDA/PyTorch downloads are several GB.
- Allow at least 25 GB of free space on the drive hosting Docker's virtual disk for the first CUDA image build, temporary files, and image export.

The image uses Python 3.12 and the project's installed inference library versions, including PyTorch 2.9.1 with CUDA 12.6. CUDA libraries come from the PyTorch wheel dependencies. The Windows `.venv` is never copied. `requirements.txt` covers inference/API use; training tools such as TRL and datasets are separate from this image. Direct dependencies are pinned; transitive dependencies and the base-image tag are not a complete immutable lock.

## Build and test

```powershell
Set-Location D:\projects\arabic-tool-llm
docker build --progress plain -t arabic-tool-llm:local .
docker build --progress plain --target test -t arabic-tool-llm:test .
```

The optional `test` target runs the existing nine tests with fake agents. It does not verify model inference. The default final `runtime` image excludes tests and pytest.

For an equivalent native dependency installation (outside Docker):

```powershell
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

## Verify GPU access

```powershell
docker run --rm --gpus all arabic-tool-llm:local python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is not available inside Docker'; print('GPU:', torch.cuda.get_device_name(0)); print('Capability:', torch.cuda.get_device_capability(0)); print(torch.ones(1, device='cuda').cpu())"
```

If this fails, fix Docker/WSL GPU access before starting the real API. No CPU fallback is configured for this 4-bit GPU setup.

## Run the real API

```powershell
Set-Location D:\projects\arabic-tool-llm
$adapterPath = (Resolve-Path .\models\qwen3-0.6b-tool-calling-v2-lora).Path
New-Item -ItemType Directory -Force .\models\hf-cache | Out-Null
$hfCachePath = (Resolve-Path .\models\hf-cache).Path
docker run -d --name arabic-tool-llm --gpus all `
  -p 127.0.0.1:8000:8000 `
  --mount "type=bind,source=$adapterPath,target=/app/models/qwen3-0.6b-tool-calling-v2-lora,readonly" `
  --mount "type=bind,source=$hfCachePath,target=/home/app/.cache/huggingface" `
  --mount "type=volume,source=arabic-tool-llm-logs,target=/app/logs" `
  arabic-tool-llm:local

docker logs -f arabic-tool-llm
```

Wait for `Application startup complete`. Press Ctrl+C to stop following logs; the detached container keeps running. The first startup downloads `Qwen/Qwen3-0.6B`; later starts reuse `models\hf-cache` on D:. If this cache is seeded from an existing local Qwen cache, the large model download is avoided. The LoRA adapter is mounted read-only at the path already expected by the application. One Uvicorn worker loads one model copy. Do not add `--reload` for this container.

For authenticated Hugging Face downloads, set `HF_TOKEN` in your PowerShell environment and add `--env HF_TOKEN` before the image name. Do not put tokens in the Dockerfile, build arguments, or committed files. The public model can also download without a token.

## Verify HTTP and model inference

```powershell
Invoke-RestMethod http://localhost:8000/health

$body = @{ message = 'فين الطلب رقم 3147 وصل؟' } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri http://localhost:8000/chat `
  -Method Post -ContentType 'application/json; charset=utf-8' `
  -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
  -TimeoutSec 180 | ConvertTo-Json -Depth 10

docker inspect --format '{{.State.Health.Status}}' arabic-tool-llm
```

Expected health response: `status: healthy`, `model_loaded: true`. The order-status request should return `get_order_status` with `order_id: 3147` and `out_for_delivery`. A 422 response can indicate an invalid model-generated tool call; inspect the logs. Valid JSON alone does not establish that the model selected the correct tool.

Swagger UI: <http://localhost:8000/docs>

The Docker health check allows a 20-minute startup grace period for the first download. A slow download can exceed this grace period; check logs. An unhealthy status does not automatically restart the container.

## Stop / restart / recreate

```powershell
docker stop arabic-tool-llm
docker start arabic-tool-llm
```

After rebuilding the image, stop and remove the old container, then repeat `docker run`:

```powershell
docker stop arabic-tool-llm
docker rm arabic-tool-llm
```

The cache folder on D: and the named log volume survive container removal. This demo's mock order data remains in memory and resets when the process restarts.

## Reference documentation

- [PyTorch CUDA 12.6 installation for 2.9.1](https://pytorch.org/get-started/previous-versions/)
- [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/)
- [bitsandbytes hardware and CUDA requirements](https://huggingface.co/docs/bitsandbytes/installation)

## Storage on this Windows machine

During setup, the first CUDA build exhausted the available space on C:. With your approval, the original Docker data disk was copied to `D:\DockerData\disk\docker_data.vhdx`, verified against the source using SHA-256, and the original C: copy was removed. Docker's existing path `C:\Users\hp\AppData\Local\Docker\wsl\disk` is now an NTFS directory junction targeting `D:\DockerData\disk`. Docker's seven pre-existing image tags and four stopped containers were visible again after recovery.

Keep `D:\DockerData` in place: it contains Docker images, volumes and build cache for all projects, not just this API. The small Docker/WSL application files and settings remain on C:. Settings backups and a separate fresh disk created during recovery were retained under `D:\DockerData`; the active original data disk is the one under `disk`. The temporary unrecognized `WslDataFolder` setting was removed.

Fixerpy helper to clone/build/run the NVIDIA Fixer container and run inference on example images.

Usage requires Docker (with GPU support for best performance) and the `uv` tool.

Quickstart
- Clone + build + download weights + run inference on Fixer examples:
  - `uv run fixer`

Details
- What `uv run fixer` does:
  - Clones `https://github.com/nv-tlabs/Fixer.git` into `.fixer_work/Fixer` (if missing)
  - Builds the Docker image from `Dockerfile.cosmos` with tag `fixer-cosmos-env`
  - Downloads model weights via `hf download nvidia/Fixer --local-dir models` into the cloned repo
  - Runs the container per Fixer README, mounting the repo to `/work` and invoking the inference script on example images:
    - `python /work/src/inference_pretrained_model.py --model /work/models/pretrained/pretrained_fixer.pkl --input /work/input --output /work/output --timestep 250`
  - Outputs are saved to `./output` (relative to current working directory)

CLI options
- `--dest-root PATH`: Working directory (default: `.fixer_work`)
- `--input PATH`: Input directory (default: `<dest-root>/Fixer/examples`)
- `--output PATH`: Output directory (default: `./output`)
- `--timestep INT`: Inference timestep (default: `250`)
- `--batch-size INT`: Batch size (default: `1`)
- `--test-speed`: Run speed benchmark before inference
- `--no-gpus`: Disable GPU pass-through for the container
- `--platform linux/amd64|linux/arm64`: Build target platform (auto-detected by default)

Examples
- Run with defaults (examples → output):
  - `uv run fixer`
- Run with custom input/output:
  - `uv run fixer --input /path/to/images --output /path/to/out --timestep 250`
- Force platform (for cross-arch hosts or when needed):
  - `uv run fixer --platform linux/amd64`

Notes
- This project uses Docker and the Fixer README’s recommended `/work` mount layout.
- Weights are downloaded to `.fixer_work/Fixer/models` via Hugging Face CLI.
- Ensure your Docker supports GPU (`--gpus all`) if you want GPU acceleration.

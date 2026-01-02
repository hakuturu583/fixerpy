from __future__ import annotations

import argparse
from pathlib import Path

from .fixer import setup_and_infer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Fixer on example images (clone/build/weights + inference)."
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path(".fixer_work"),
        help="Working directory for cloning and outputs (default: .fixer_work)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input images directory (default: <dest-root>/Fixer/examples)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <dest-root>/output)",
    )
    parser.add_argument("--timestep", type=int, default=250, help="Inference timestep")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--no-gpus",
        action="store_true",
        help="Disable GPU pass-through for the container",
    )
    parser.add_argument(
        "--test-speed",
        action="store_true",
        help="Run speed benchmark before inference",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Docker build platform (e.g., linux/amd64, linux/arm64). If omitted, auto-detects host.",
    )

    args = parser.parse_args()

    dest_root: Path = args.dest_root
    default_input = dest_root / "Fixer" / "examples"
    input_dir: Path | None = args.input if args.input else default_input
    # Default output to ./output under current working directory
    output_dir: Path | None = args.output if args.output else (Path.cwd() / "output")

    setup_and_infer(
        dest_root=dest_root,
        input_dir=input_dir,
        output_dir=output_dir,
        timestep=args.timestep,
        batch_size=args.batch_size,
        test_speed=args.test_speed,
        use_gpus=not args.no_gpus,
        platform=args.platform,
    )


if __name__ == "__main__":
    main()

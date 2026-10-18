import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="TraceLens research and local incident explorer")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download the original BGL archive from Loghub/Zenodo")
    fetch.add_argument("--output", type=Path, default=Path("data/BGL.zip"))
    train = commands.add_parser("train", help="Fit models and export reproducible evaluation artifacts")
    train.add_argument("--data", type=Path, default=Path("data/BGL.zip"))
    train.add_argument("--config", type=Path, default=Path("configs/portfolio.json"))
    train.add_argument("--output", type=Path, default=Path("artifacts/portfolio"))
    train.add_argument("--max-lines", type=int)
    serve = commands.add_parser("serve", help="Run the dashboard and real checkpoint inference locally")
    serve.add_argument("--artifacts", type=Path, default=None)
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "download":
        from .data import download
        download(args.output)
        print(f"Downloaded {args.output}")
    elif args.command == "train":
        from .train import run
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.max_lines is not None:
            config["max_lines"] = args.max_lines
        if config["max_lines"] < 100 or config["window_size"] < 3:
            parser.error("Use at least 100 lines and window_size >= 3.")
        run(args.data, args.output, config)
    else:
        import uvicorn
        from .api import create_app
        artifacts=args.artifacts or (Path('artifacts/validated') if Path('artifacts/validated/report.json').exists() else Path('artifacts/portfolio'))
        uvicorn.run(create_app(artifacts), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

import json

from src.config import build_config, parse_args
from src.pipeline import DeliveryMLPipeline


def main():
    args = parse_args()
    config = build_config(args)
    summary = DeliveryMLPipeline(config).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier.")

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="./../datasets/visa_finetune/",
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default="results", help="Output directory"
    )
    parser.add_argument(
        "--model-name", "-m", type=str, default="resnet18", help="Model name (timm)"
    )
    parser.add_argument(
        "--img-size", "-i", default=(224, 224), help="Input size of image"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--save-interval", "-s", type=int, default=1, help="Save interval (epoch)"
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Batch size")

    parser.add_argument(
        "--num-workers", "-w", type=int, default=10, help="Number of workers"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--gpu-ids", type=int, default=None, nargs="+", help="GPU IDs to use"
    )
    group.add_argument("--n-gpu", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--debug", type=bool, default=False, help="Runs debug mode")
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    args = get_args()
    print(args)

import argparse
import numpy as np
import torch
import yaml
from helper import (
    accuracy,
    generate_weights,
    load_precomputed_features_bifta,
    set_seed,
    create_random_crop_loader,
    create_grid_crop_loader,
    load_huggingface_models,
)
from clip import clip
from torch.nn import functional as F


def parse_args():
    """Parse command line arguments with clear documentation."""
    parser = argparse.ArgumentParser(
        description="BiFTA: Bi-level Feature Augmentation for Test-time Adaptation"
    )

    # Dataset and basic config
    parser.add_argument(
        "--dataset", type=str, default="imagenet",
        help="Dataset name (e.g., imagenet, cub, food101)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size for data loading (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (cuda or cpu)"
    )

    # Augmentation parameters (override config if specified)
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Minimum crop scale factor (overrides config)"
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Maximum crop scale factor (overrides config)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=None,
        help="Number of augmented samples to generate (overrides config)"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=None,
        help="IoU threshold for filtering crops (overrides config)"
    )
    parser.add_argument(
        "--max_queue_len", type=int, default=None,
        help="Maximum queue length for filtered crops (overrides config)"
    )
    parser.add_argument(
        "--model_size", type=str, default=None,
        help="CLIP visual encoder name."
    )

    # Crop strategy parameters
    parser.add_argument(
        "--crop_strategy", type=str, default="random", choices=["random", "grid"],
        help="Crop strategy: 'random' (random crops with IoU filtering) or 'grid' (deterministic grid crops)"
    )
    parser.add_argument(
        "--num_grid", type=int, default=3,
        help="Grid size for grid_crop strategy (e.g., 3 for 3x3 grid). Only used when crop_strategy='grid'"
    )

    # Ablation study parameters
    parser.add_argument(
        "--ablation_mode", type=int, default=0, choices=[0, 1],
        help="0: normal mode, 1: ablation mode (saves features with all hyperparams in filename)"
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Setup device
    device = torch.device(args.device)
    print("=" * 80)
    print(f"BiFTA: Bi-level Feature Augmentation for Test-time Adaptation")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Num workers: {args.num_workers}")
    print(f"Random seed: {args.seed}")

    # Load config file
    config_path = f"configs/{args.dataset}.yaml"
    print(f"Loading config: {config_path}")
    with open(file=config_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    set_seed(args.seed)

    # Load hyperparameters with priority: CLI args > config file > defaults
    # Basic parameters from config
    model_size = args.model_size if args.model_size is not None else hparams["model_size"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]

    # Augmentation parameters (allow CLI override)
    alpha = args.alpha if args.alpha is not None else hparams.get("alpha", 0.5)
    beta = args.beta if args.beta is not None else hparams.get("beta", 0.9)
    n_samples = args.n_samples if args.n_samples is not None else hparams["n_samples"]
    iou_threshold = args.iou_threshold if args.iou_threshold is not None else hparams["iou_threshold"]
    max_queue_len = args.max_queue_len if args.max_queue_len is not None else hparams["max_queue_len"]

    # Print augmentation parameters
    print("\nCrop Strategy:")
    print(f"  strategy: {args.crop_strategy}")
    if args.crop_strategy == "random":
        print(f"  alpha (min crop scale): {alpha}")
        print(f"  beta (max crop scale): {beta}")
        print(f"  n_samples: {n_samples}")
        print(f"  iou_threshold: {iou_threshold}")
        print(f"  max_queue_len: {max_queue_len}")
    else:  # grid
        print(f"  num_grid: {args.num_grid}")
        print(f"  total_crops: {args.num_grid * args.num_grid}")
    print(f"  ablation_mode: {args.ablation_mode}")
    print("=" * 80)


    # load model
    clip_model_dict = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN-50", "RN-101"]
    if model_size not in clip_model_dict:
        print(f"Loading {model_size}")
        model, processor = load_huggingface_models(model_name=model_size, device="cuda")
    else:
        print(f"Loading {model_size}")
        model, processor = clip.load(model_size, device=device)

    model.eval()
    model.requires_grad_(False)

    # Create custom loader based on crop strategy
    if args.crop_strategy == "random":
        custom_loader = create_random_crop_loader(
            processor=processor,
            n_samples=n_samples,
            alpha=alpha,
            beta=beta,
            iou_threshold=iou_threshold,
            max_queue_len=max_queue_len
        )
        crop_config = {
            "strategy": "random",
            "alpha": alpha,
            "beta": beta,
            "n_samples": n_samples,
            "iou_threshold": iou_threshold,
            "max_queue_len": max_queue_len,
        }
    elif args.crop_strategy == "grid":
        custom_loader = create_grid_crop_loader(
            processor=processor,
            num_grid=args.num_grid
        )
        crop_config = {
            "strategy": "grid",
            "num_grid": args.num_grid,
        }
    else:
        raise ValueError(f"Unknown crop strategy: {args.crop_strategy}")

    # Pre-compute and load visual features
    # Returns:
    #   - patch_embeddings_with_weights: [B, N, D+1] patch features with similarity weights
    #   - labels: [B] ground truth labels
    #   - global_image_features: [B, D] full image features (before cropping)
    #   - num_valid_crops: [B] number of real crops per image (excluding zero padding)
    (
        patch_embeddings_with_weights,
        labels,
        global_image_features,
        num_valid_crops,
    ) = load_precomputed_features_bifta(
        model,
        dataset_name=args.dataset,
        model_size=model_size,
        crop_config=crop_config,
        batch_size=batch_size,
        num_workers=args.num_workers,
        data_path=data_path,
        custom_loader=custom_loader,
        device=device,
        ablation_mode=args.ablation_mode,
    )

    num_patches = patch_embeddings_with_weights.size(1)
    global_image_features = global_image_features.to(device)
    num_valid_crops = num_valid_crops.to(device)

    # Print statistics about crop filtering
    avg_valid_crops = num_valid_crops.float().mean().item()
    min_valid_crops = num_valid_crops.min().item()
    max_valid_crops = num_valid_crops.max().item()
    zero_padding_ratio = (num_patches - avg_valid_crops) / num_patches * 100

    print("\n" + "=" * 80)
    print("Crop Filtering Statistics:")
    print(f"  Average valid crops per image: {avg_valid_crops:.1f} / {num_patches}")
    print(f"  Min valid crops: {min_valid_crops}")
    print(f"  Max valid crops: {max_valid_crops}")
    print(f"  Average zero-padding ratio: {zero_padding_ratio:.1f}%")
    print("=" * 80)

    results = {}
    with torch.no_grad():
        methods = hparams["methods"]
        for method in methods:
            method = list(method.values())[0]
            method_name = method["name"]
            method_enabled = method["enabled"]

            text_scale = (
                torch.exp(torch.tensor(method["text_scale"])).to(device)
                if "text_scale" in method
                else None
            )
            image_scale = (
                torch.exp(torch.tensor(method["image_scale"])).to(device)
                if "image_scale" in method
                else None
            )

            if method_enabled:
                zeroshot_weights = generate_weights(
                    method_name,
                    model=model,
                    dataset_name=args.dataset,
                    tt_scale=text_scale,
                    device=device,
                )
                # Set zero-shot weights to the same dtype as image features
                zeroshot_weights = zeroshot_weights.to(global_image_features.dtype)
            else:
                continue

            # Baseline: evaluate using global image features (no augmentation)
            logits = global_image_features.squeeze(1) @ zeroshot_weights
            baseline_acc = accuracy(
                logits, labels, global_image_features.size(0), args.dataset
            )
            if method_name not in ["wca", "bifta"]:
                print(f"{method_name}: {baseline_acc:.2f}\n")
                results[method_name] = round(baseline_acc, 2)

            else:
                acc_list = []
                patch_num = hparams["patch_n"]
                print(f"\nRunning BiFTA with {hparams['n_run']} runs, {patch_num} patches per run")

                for run_idx in range(hparams["n_run"]):
                    # Randomly sample patches from the pool
                    # Sample from valid crops only (excluding zero padding)
                    batch_size = patch_embeddings_with_weights.size(0)
                    sampled_patches_list = []

                    for i in range(batch_size):
                        valid_num = num_valid_crops[i].item()
                        # Sample with replacement from valid crops only
                        random_indices = torch.randint(0, valid_num, (patch_num,), device=device)
                        sampled_patches_list.append(patch_embeddings_with_weights[i, random_indices, :])

                    sampled_patches_with_weights = torch.stack(sampled_patches_list)

                    # Split into patch embeddings and their weights
                    # Shape: [B, patch_num, D] and [B, patch_num]
                    patch_embeds = sampled_patches_with_weights[:, :, :-1]
                    patch_similarity_weights = sampled_patches_with_weights[:, :, -1]
                    del sampled_patches_with_weights

                    # Compute weighted average of patch embeddings
                    # w_i: [B, patch_num, 1] - normalized weights for each patch
                    w_i = (patch_similarity_weights * image_scale).softmax(-1).unsqueeze(-1)
                    aggregated_features = (patch_embeds * w_i).sum(dim=1)  # [B, D]
                    aggregated_features = F.normalize(aggregated_features, dim=-1)

                    # Compute logits: [B, D] @ [C, D].T -> [B, C]
                    logits = aggregated_features @ zeroshot_weights
                    acc_list.append(
                        accuracy(logits, labels, aggregated_features.size(0), args.dataset)
                    )

                mean_acc = np.mean(acc_list)
                std_acc = np.std(acc_list)
                print(f"\n{method_name}: {mean_acc:.2f} ± {std_acc:.2f}")
                print(f"Individual runs: {[f'{acc:.2f}' for acc in acc_list]}")
                results[method_name] = {"mean": round(mean_acc, 2), "std": round(std_acc, 2)}


if __name__ == "__main__":
    main()

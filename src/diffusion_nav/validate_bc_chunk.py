import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import Normalizer, SequenceDataset
from .models import BC_Chunk
from .training import choose_device, load_config, validate_bc_chunk


def validate_saved_bc_chunk(config: dict, checkpoint_path: str | Path | None = None) -> dict:
    if config["model"].get("type") != "BC_Chunk":
        raise ValueError("the configuration must select model type BC_Chunk")

    data_config = config["data"]
    training_config = config["training"]
    validation_path = Path(data_config["validation_path"])
    checkpoint_path = Path(checkpoint_path or config["output"]["checkpoint_path"])
    if not validation_path.is_file():
        raise FileNotFoundError(f"validation dataset not found: {validation_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BC-Chunk checkpoint not found: {checkpoint_path}")

    device = choose_device(str(training_config.get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("policy_type") != "bc_chunk":
        raise ValueError("checkpoint doesn't contain a BC-Chunk policy")
    for key in ("model_config", "model_state_dict", "normalizer", "epoch", "metrics"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing {key}")
    if "validation_loss" not in checkpoint["metrics"]:
        raise ValueError("checkpoint metrics are missing validation_loss")

    model = BC_Chunk(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    validation_dataset = SequenceDataset(validation_path, model.obs_horizon, model.pred_horizon, normalizer=normalizer)
    if validation_dataset.observations.shape[1] != model.obs_dim:
        raise ValueError("validation observation dimension doesn't match checkpoint")

    batch_size = int(data_config.get("batch_size", 64))
    num_workers = int(data_config.get("num_workers", 0))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    validation_loss = validate_bc_chunk(model, validation_loader, device)

    return {"metric": "normalized_action_mse", "validation_loss": validation_loss,
            "checkpoint_validation_loss": float(checkpoint["metrics"]["validation_loss"]), "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_path": str(checkpoint_path), "validation_path": str(validation_path),
            "sample_count": len(validation_dataset), "episode_count": len(validation_dataset.episode_ends),
            "map_seeds": sorted({int(seed) for seed in validation_dataset.map_seeds}), "obs_horizon": model.obs_horizon,
            "pred_horizon": model.pred_horizon, "device": str(device)}


def save_validation_result(result: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute BC-Chunk validation loss.")
    parser.add_argument("--config", default="configs/bc_chunk.yaml")
    parser.add_argument("--checkpoint", help="Override output.checkpoint_path from the config")
    parser.add_argument("--output", default="artifacts/bc_chunk_validation.json")
    arguments = parser.parse_args()

    result = validate_saved_bc_chunk(load_config(arguments.config), arguments.checkpoint)
    output_path = save_validation_result(result, arguments.output)
    print(f"validation_loss ({result['metric']}): {result['validation_loss']:.6f}")
    print(f"checkpoint epoch: {result['checkpoint_epoch']}")
    print(f"saved validation_loss: {result['checkpoint_validation_loss']:.6f}")
    print(f"samples: {result['sample_count']} | episodes: {result['episode_count']}")
    print(f"saved report: {output_path}")


if __name__ == "__main__":
    main()

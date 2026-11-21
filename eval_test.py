import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from models.decoder_pst import EPSTDecoder
from models.mrsother import MrsOther3DFinetune

ARCHIVE_DIR = Path("checkpoint/D_ablation_EPST")
ENCODER_DIR = Path("checkpoint/D_train_mrs_3D")

def encoder_for(checkpoint) -> Path:
    recorded = Path(checkpoint["encoder_checkpoint"])
    return recorded if recorded.is_file() else ENCODER_DIR / recorded.name


def load_encoder_weights(model: MrsOther3DFinetune, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    prefixes = ("patch_embed", "cls_token", "pos_embed", "time_pos_embed",
                "space_pos_embed", "blocks", "norm", "input_proj")
    encoder_state = model.encoder.state_dict()
    filtered = {
        key: value for key, value in state.items()
        if key.startswith(prefixes) and key in encoder_state
        and encoder_state[key].shape == value.shape
    }
    model.encoder.load_state_dict(filtered, strict=False)


def build_model(encoder_path: Path, decoder_path: Path, device: torch.device):
    decoder = EPSTDecoder()
    model = MrsOther3DFinetune(decoder).to(device)
    load_encoder_weights(model, encoder_path)
    checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=False)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"], strict=True)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, float(checkpoint["label_mean"]), float(checkpoint["label_std"])


@torch.no_grad()
def infer(model, image: np.ndarray, valid: np.ndarray, mask: np.ndarray,
          label_mean: float, label_std: float, device: torch.device, batch_size: int = 32):
    predictions = []
    for start in range(0, len(image), batch_size):
        stop = start + batch_size
        output = model(
            torch.from_numpy(image[start:stop]).to(device),
            torch.from_numpy(valid[start:stop]).to(device),
        )
        anchor = torch.from_numpy(mask[start:stop] > 0).to(device)
        predictions.append((output[anchor] * label_std + label_mean).cpu().numpy())
    return np.concatenate(predictions)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    residual = prediction - target
    return {
        "n": int(target.size),
        "r2": float(1.0 - (residual ** 2).sum() / ((target - target.mean()) ** 2).sum()),
        "rmse": float(np.sqrt((residual ** 2).mean())),
        "mae": float(np.abs(residual).mean()),
    }


def main() -> int:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    archives = sorted(ARCHIVE_DIR.glob("decoder_dynamic_routing_outer*_best.pth"))

    print(f"device: {device}")
    print(f"{'fold':>4} {'RegionIndex':>11} {'pred(kg/plot)':>14} {'label(kg/plot)':>15}")

    predictions, targets, regions = [], [], []
    for archive in archives:
        fold = int(archive.stem.split("_outer")[1].split("_")[0])
        checkpoint = torch.load(archive, map_location="cpu", weights_only=False)
        data = np.load(f"Data/fold_{fold}.npz")
        model, label_mean, label_std = build_model(encoder_for(checkpoint), archive, device)
        pred = infer(model, data["image"], data["valid"], data["label_mask"],
                     label_mean, label_std, device)
        for region, value, truth in zip(data["region"], pred, data["label"]):
            print(f"{fold:>4} {int(region):>11} {value:>14.2f} {float(truth):>15.2f}")
        predictions.append(pred)
        targets.append(data["label"].astype(np.float64))
        regions.append(data["region"])

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    region = np.concatenate(regions)
    result = metrics(prediction, target)

    np.savetxt("predictions.csv", np.column_stack([region, prediction, target]),
               delimiter=",", header="RegionIndex,pred,label", comments="", fmt="%d,%.6f,%.6f")

    print("-" * 50)
    print(f"n={result['n']}  R²={result['r2']:.6f}  "
          f"RMSE={result['rmse']:.2f} kg/plot ({result['rmse'] / 40:.2f} Mg/ha)  "
          f"MAE={result['mae']:.2f} kg/plot ({result['mae'] / 40:.2f} Mg/ha)")

if __name__ == "__main__":
    main()

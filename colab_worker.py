import os
import glob
import time
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, List, Optional, Tuple
import requests

# Self-healing import block to fetch hpo_client.py dynamically from the broker if not present locally
try:
    from src.hpo_client import TrialSession
except ImportError:
    try:
        from hpo_client import TrialSession
    except ImportError:
        print("hpo_client.py not found locally. Attempting to download from broker...")
        # Resolve broker URL from environment
        broker_url = os.getenv("HPO_BROKER_URL")
        if not broker_url:
            raise ValueError(
                "Neither src.hpo_client nor hpo_client could be imported, "
                "and HPO_BROKER_URL is not set to download hpo_client.py from the broker."
            )
        try:
            headers = {"ngrok-skip-browser-warning": "1"}
            token = os.getenv("HPO_SECRET_TOKEN")
            if token:
                headers["X-HPO-Token"] = token
            r = requests.get(
                f"{broker_url.rstrip('/')}/hpo_client.py",
                headers=headers,
                timeout=15,
            )
            r.raise_for_status()
            with open("hpo_client.py", "w") as f:
                f.write(r.text)
            print("Successfully downloaded hpo_client.py from broker.")
            from hpo_client import TrialSession
        except Exception as e:
            print(f"Error downloading hpo_client.py from broker: {e}")
            raise

# --- 1. YOUR EXACT DATASET CLASS (EMBEDDED FOR EASY COLAB USE) ---
class CrackDataset(Dataset):
    """
    PyTorch Dataset representing bridge defect imagery
    and annotated segmentation binary masks.
    """
    def __init__(self, img_paths: list, mask_paths: list, transform: A.Compose = None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> tuple:
        # Load imagery and convert to RGB channels
        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load annotation mask in grayscale
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Retrieve binary values from grayscale mask
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # INVERSION: DeepCrack uses black (0) for cracks and white (1) for background.
        # We invert it so cracks become 1 and background becomes 0 to match your model's expectations.
        mask = 1 - mask

        mask = np.expand_dims(mask, axis=-1).astype(np.float32)

        # Match image and mask transformations
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        return img, mask


# Augmentation using your albumentations recipes, adapted for dynamic resolutions
def get_train_transform(resolution: int) -> A.Compose:
    return A.Compose([
        A.Resize(resolution, resolution),  # Injected dynamically for HPO tuning
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_test_transform(resolution: int) -> A.Compose:
    """
    Deterministic normalization transform for validation/test crack detection.
    """
    return A.Compose([
        A.Resize(resolution, resolution),  # Injected dynamically for HPO tuning
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# --- 2. DATASET DOWNLOAD & EXTRACTION ---
# Automatically downloads and unzips DeepCrack if it is not present in /content/input/
if not os.path.exists("/content/input/DeepCrack") and not os.path.exists("/content/DeepCrack"):
    print("Downloading DeepCrack dataset from GitHub...")
    os.makedirs("/content/input", exist_ok=True)
    # Using raw download URL from yhlleo/DeepCrack
    os.system("wget -q -O /content/input/DeepCrack.zip https://github.com/yhlleo/DeepCrack/raw/master/dataset/DeepCrack.zip")
    print("Extracting DeepCrack.zip to /content/input/...")
    os.system("unzip -o -q /content/input/DeepCrack.zip -d /content/input/")
    print("DeepCrack dataset downloaded and extracted successfully!")

# --- 3. MODEL FETCHING SETUPS ---
# Clones your repository if not already cloned in Google Colab
if not os.path.exists("/content/crack-seg"):
    os.system("git clone https://github.com/Ishaan1402/crack-seg.git /content/crack-seg")

import sys
sys.path.append("/content/crack-seg")

from src.models.unet import UNet  # Imports your real UNet model definition

# --- 4. HPO BROKER CONFIGURATION ---
BROKER_URL = os.getenv("HPO_BROKER_URL")
if not BROKER_URL:
    raise ValueError("HPO_BROKER_URL environment variable must be set to connect to Pathfinder.")

# ngrok free tier: skip interstitial; always POST JSON to API paths
BROKER_HTTP_HEADERS = {
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "1",
}

def broker_get(path: str, timeout: int = 30):
    base = BROKER_URL.rstrip("/")
    for suffix in ("/api/suggest_trial", "/api/suggest_trials", "/api"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    if not path.startswith("/"):
        path = "/" + path
    return requests.get(
        base + path,
        headers={"ngrok-skip-browser-warning": "1"},
        timeout=timeout,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 6. YOUR SLIDING WINDOW NORMALIZATION CODE INTEGRATION ---
def run_sliding_window_inference(model: nn.Module, image: torch.Tensor, patch_size=448, overlap=0.5, t_val=0.5):
    """
    Tiled sliding window inference combining overlapping predictions with a 2D Gaussian weight map.
    """
    model.eval()
    c, h_img, w_img = image.shape

    accum_p = np.zeros((h_img, w_img), dtype=np.float32)
    accum_w = np.zeros((h_img, w_img), dtype=np.float32)
    stride = int(patch_size * (1.0 - overlap))

    gaussian_patch = np.outer(np.hamming(patch_size), np.hamming(patch_size)).astype(np.float32)

    with torch.no_grad():
        for y in range(0, h_img - patch_size + 1, stride):
            for x in range(0, w_img - patch_size + 1, stride):
                # Extract image patch
                patch = image[:, y : y + patch_size, x : x + patch_size].unsqueeze(0).to(device)
                logits = model(patch)
                probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()

                accum_p[y : y + patch_size, x : x + patch_size] += probs * gaussian_patch
                accum_w[y : y + patch_size, x : x + patch_size] += gaussian_patch

        # Element-wise division to normalize final blended probabilities
        final_probs = accum_p / np.maximum(accum_w, 1e-5)
        binary_mask = (final_probs > t_val).astype(np.uint8)
        crack_area_ratio = float(np.sum(binary_mask) / (h_img * w_img))

        return final_probs, binary_mask, crack_area_ratio


DEFAULT_HPO_CONFIG: Dict[str, Any] = {
    "eval_protocol": {
        "enabled": False,
        "fixed_resolution": None,
        "train_resolution_param": "resolution",
        "fixed_dice_attr": "dice_eval_fixed",
        "fixed_bce_attr": "bce_eval_fixed",
        "use_fixed_metric_for_pruning": False,
        "patch_size_below_512": 256,
        "patch_size_at_512_plus": 448,
    },
    "legacy_param_aliases": {"encoder_name": "model_capacity"},
    "legacy_capacity_values": {
        "resnet34": "narrow",
        "efficientnet-b0": "narrow",
        "resnet50": "wide",
    },
}


def fetch_hpo_config() -> Dict[str, Any]:
    if BROKER_URL:
        try:
            resp = broker_get("/api/hpo_config", timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"Warning: could not fetch hpo_config from broker ({exc}); using defaults.")
    return json.loads(json.dumps(DEFAULT_HPO_CONFIG))


def normalize_trial_params(params: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    for old, new in config.get("legacy_param_aliases", {}).items():
        if old in out and new not in out:
            val = out.pop(old)
            mapped = config.get("legacy_capacity_values", {}).get(val, val)
            out[new] = mapped
    return out


def unet_features_from_params(params: Dict[str, Any], config: Dict[str, Any]) -> List[int]:
    norm = normalize_trial_params(params, config)
    capacity = norm.get("model_capacity", "narrow")
    if capacity == "wide":
        return [64, 128, 256, 512]
    return [32, 64, 128, 256]


def patch_size_for_resolution(resolution: int, config: Dict[str, Any]) -> int:
    ev = config.get("eval_protocol", {})
    if resolution < 512:
        return int(ev.get("patch_size_below_512", 256))
    return int(ev.get("patch_size_at_512_plus", 448))


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice


def run_validation_epoch(
    model: nn.Module,
    val_images: List[str],
    val_masks: List[str],
    resolution: int,
    bce_loss_fn: nn.Module,
    config: Dict[str, Any],
) -> Tuple[float, float]:
    val_dataset = CrackDataset(
        img_paths=val_images,
        mask_paths=val_masks,
        transform=get_val_test_transform(resolution),
    )
    model.eval()
    val_dice_list = []
    val_bce_list = []
    patch_size = patch_size_for_resolution(resolution, config)

    for val_img, val_mask in val_dataset:
        final_probs, binary_mask, _ = run_sliding_window_inference(
            model=model,
            image=val_img,
            patch_size=patch_size,
            overlap=0.5,
            t_val=0.5,
        )
        target_np = val_mask.numpy().astype(np.uint8)
        if target_np.ndim == 3:
            target_np = np.squeeze(target_np)
        val_dice_list.append(calculate_dice_score(binary_mask, target_np))

        probs_t = torch.tensor(final_probs).unsqueeze(0).unsqueeze(0)
        if val_mask.dim() == 3 and val_mask.shape[2] == 1:
            val_mask = val_mask.permute(2, 0, 1)
        target_for_loss = val_mask.unsqueeze(0)
        val_loss = bce_loss_fn(
            torch.logit(torch.clamp(probs_t, 1e-6, 1 - 1e-6)), target_for_loss
        )
        val_bce_list.append(val_loss.item())

    return float(np.mean(val_dice_list)), float(np.mean(val_bce_list))


def calculate_dice_score(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    intersection = np.sum(pred_mask * target_mask)
    union = np.sum(pred_mask) + np.sum(target_mask)
    if union == 0:
        return 1.0
    return float((2. * intersection) / union)


# --- 7. MAIN HPO RUNNER FOR COLAB ---
COLAB_WORKER_REV = "2025-06-07-2"  # bump after broker-side edits; Colab should re-fetch /colab_worker.py

#
# Public entrypoints (bridge-crack reference only):
#   train_colab_trial       — run one trial (suggest → train → complete / prune / fail)
#   train_colab_trial_loop  — call train_colab_trial N times; survives guardrail skips,
#                             caught OOM, and transient suggest errors; clears CUDA cache
#                             between iterations.


def train_colab_trial(study_name: str, epochs=15):
    """Run a single HPO trial on Colab (one suggest → complete cycle)."""
    # Ensure folder for saved checkpoints exists
    os.makedirs("checkpoints", exist_ok=True)

    # Detect GPU hardware telemetry if torch is available
    gpu_model = "CPU"
    max_vram_gb = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_model = torch.cuda.get_device_name(0)
            max_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except ImportError:
        pass

    session = TrialSession(broker_url=BROKER_URL, study_name=study_name)
    print(f"colab_worker rev {COLAB_WORKER_REV}")
    print(session.health())

    try:
        trial = session.suggest()
    except Exception as exc:
        print(f"ERROR: could not get suggestion from HPO broker: {exc}")
        return

    trial_id = trial["trial_id"]
    trial_number = trial.get("trial_number", trial_id)
    params = trial["params"]
    trial_label = f"#{trial_number}"

    print(f"\n--- Starting HPO Trial {trial_label} on Colab GPU (broker id={trial_id}) ---")

    def _fail_trial(epoch: int = 0, oom: bool = False, reason: str = ""):
        if reason:
            print(reason)
        try:
            session.complete(
                epoch,
                0.0,
                999.0,
                state="FAIL",
                gpu_model=gpu_model,
                max_vram_gb=max_vram_gb,
                oom_triggered=oom,
            )
            print(f"  >> Trial {trial_label} reported as FAILED to broker.")
        except Exception as report_err:
            print(f"  >> Could not report failure to broker: {report_err}")

    try:
        hpo_config = fetch_hpo_config()
        print(f"Parameters: {params}")

        required = ["learning_rate", "batch_size", "resolution", "model_capacity", "loss_weight_ratio"]
        missing = [k for k in required if k not in params]
        if missing:
            _fail_trial(reason=f"ERROR: Trial {trial_id} has incomplete hyperparameters (missing: {missing}).")
            return

        # Pre-flight guardrail (predictable VRAM blow-up; not a caught CUDA OOM)
        if params.get("resolution", 512) == 1024 and params.get("batch_size", 8) >= 16:
            _fail_trial(oom=True, reason="Guardrail: resolution 1024 with batch_size >= 16 — skipping training.")
            return

        lr = params.get("learning_rate", 1e-3)
        batch_size = int(params.get("batch_size", 8))
        resolution = int(params.get("resolution", 512))
        loss_weight_ratio = params.get("loss_weight_ratio", 0.5)

        unet_features = unet_features_from_params(params, hpo_config)
        ev = hpo_config.get("eval_protocol", {})

        DEEPCRACK_DIR = "/content/input"
        if os.path.exists(f"{DEEPCRACK_DIR}/DeepCrack/train_img"):
            DEEPCRACK_DIR = f"{DEEPCRACK_DIR}/DeepCrack"

        print(f"Loading DeepCrack images from: {DEEPCRACK_DIR}")

        train_images = sorted(glob.glob(f"{DEEPCRACK_DIR}/train_img/*.jpg") +
                              glob.glob(f"{DEEPCRACK_DIR}/train_img/*.png"))
        train_masks = sorted(glob.glob(f"{DEEPCRACK_DIR}/train_lab/*.jpg") +
                             glob.glob(f"{DEEPCRACK_DIR}/train_lab/*.png"))

        val_images = sorted(glob.glob(f"{DEEPCRACK_DIR}/test_img/*.jpg") +
                            glob.glob(f"{DEEPCRACK_DIR}/test_img/*.png"))
        val_masks = sorted(glob.glob(f"{DEEPCRACK_DIR}/test_lab/*.jpg") +
                           glob.glob(f"{DEEPCRACK_DIR}/test_lab/*.png"))

        if not train_images or not train_masks:
            raise FileNotFoundError(
                f"Could not find training images or masks in {DEEPCRACK_DIR}! "
                f"(Checked subfolders train_img/ and train_lab/)"
            )

        train_dataset = CrackDataset(
            img_paths=train_images,
            mask_paths=train_masks,
            transform=get_train_transform(resolution)
        )
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        model = UNet(in_channels=3, out_channels=1, features=unet_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        bce_loss_fn = nn.BCEWithLogitsLoss()

        pruned = False

        for epoch in range(1, epochs + 1):
            t_start = time.time()
            model.train()
            train_bce = 0.0

            for images, targets in dataloader:
                images, targets = images.to(device), targets.to(device)
                if targets.dim() == 4 and targets.shape[3] == 1:
                    targets = targets.permute(0, 3, 1, 2)
                optimizer.zero_grad()
                logits = model(images)
                bce = bce_loss_fn(logits, targets)
                dice_component = soft_dice_loss(logits, targets)
                lw = float(loss_weight_ratio)
                loss = lw * bce + (1.0 - lw) * dice_component
                loss.backward()
                optimizer.step()
                train_bce += loss.item() * images.size(0)

            train_bce /= len(dataloader.dataset)

            mean_dice, mean_bce = run_validation_epoch(
                model, val_images, val_masks, resolution, bce_loss_fn, hpo_config
            )

            dice_eval_fixed = None
            bce_eval_fixed = None
            fixed_res = ev.get("fixed_resolution")
            if ev.get("enabled") and fixed_res is not None:
                fixed_res = int(fixed_res)
                if fixed_res != resolution:
                    dice_eval_fixed, bce_eval_fixed = run_validation_epoch(
                        model, val_images, val_masks, fixed_res, bce_loss_fn, hpo_config
                    )
                    print(
                        f"  Fixed eval @{fixed_res}px | BCE: {bce_eval_fixed:.4f} | Dice: {dice_eval_fixed:.4f}"
                    )
                else:
                    dice_eval_fixed, bce_eval_fixed = mean_dice, mean_bce

            t_elapsed = time.time() - t_start
            total_images_processed = len(dataloader.dataset) + len(val_images)
            speed_ips = total_images_processed / t_elapsed if t_elapsed > 0 else 0.0

            gpu_memory = 0.0
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)

            print(f"  Epoch {epoch:02d} | Train BCE: {train_bce:.4f} | Val BCE: {mean_bce:.4f} | Val Dice: {mean_dice:.4f} | GPU Mem: {gpu_memory:.1f}MB | Speed: {speed_ips:.1f} img/s")

            should_prune = session.report_epoch(
                epoch,
                mean_dice,
                mean_bce,
                dice_eval_fixed=dice_eval_fixed,
                bce_eval_fixed=bce_eval_fixed,
                gpu_memory=gpu_memory,
                speed_ips=speed_ips
            )

            if should_prune:
                print(f"  >> Trial {trial_label} performing poorly. PRUNING at epoch {epoch}!")
                session.complete(
                    epoch,
                    mean_dice,
                    mean_bce,
                    dice_eval_fixed=dice_eval_fixed,
                    bce_eval_fixed=bce_eval_fixed,
                    state="PRUNED",
                    gpu_model=gpu_model,
                    max_vram_gb=max_vram_gb,
                    oom_triggered=False
                )
                pruned = True
                break

        if not pruned:
            weights_path = f"checkpoints/trial_{trial_id}_unet_res_{resolution}.pt"
            torch.save(model.state_dict(), weights_path)

            session.complete(
                epoch,
                mean_dice,
                mean_bce,
                weights_path=weights_path,
                dice_eval_fixed=dice_eval_fixed,
                bce_eval_fixed=bce_eval_fixed,
                state="COMPLETE",
                gpu_model=gpu_model,
                max_vram_gb=max_vram_gb,
                oom_triggered=False
            )
            print(f"  >> Trial {trial_label} marked as COMPLETED successfully!")

    except Exception as exc:
        oom = type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()
        if oom:
            _fail_trial(oom=True, reason=f"CUDA OOM during trial {trial_label}: {exc}")
        else:
            _fail_trial(reason=f"Trial {trial_label} crashed: {exc}")
    finally:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def train_colab_trial_loop(study_name: str, n_trials: int = 12, epochs: int = 15):
    """Run ``train_colab_trial`` repeatedly — the usual Colab entrypoint for a full study session."""
    for i in range(n_trials):
        print(f"\n========== Colab HPO iteration {i + 1}/{n_trials} ==========")
        train_colab_trial(study_name, epochs=epochs)


if __name__ == "__main__":
    train_colab_trial_loop("bridge_crack_study", n_trials=12, epochs=15)

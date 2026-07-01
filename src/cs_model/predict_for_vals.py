import argparse
import torch
import nibabel as nib
from pathlib import Path
import os

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
)

from unet import build_model
from dataset import get_subject_features
from train import load_config

VAL_DIR = "/SAN/medic/BIC-MAC-CS/bic-mac-data/val"
PATCH_SIZE = (128, 128, 128)
SW_BATCH = 1 # You can increase this if to speed up inference at the cost of VRAM
OVERLAP = 0.5


def predict():
    cfg = load_config()
    out= cfg["output_dir"]
    model_path = f"{out}/checkpoints/best_model.pth"
    transforms = Compose(
        [
            LoadImaged(
                keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase"]
            ),
            EnsureChannelFirstd(
                keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase"]
            ),
            NormalizeIntensityd(
                keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase"],
                nonzero=True,
                channel_wise=True,
                subtrahend=[0],
            ),
            ConcatItemsd(
                keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase"],
                name="input",
            ),
            EnsureTyped(keys=["input"]),
        ]
    )

    device = "cuda"

    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    for sub in sorted(os.listdir(VAL_DIR)):
        print(f"Getting subject {sub}")
        subject = get_subject_features(os.path.join(VAL_DIR, sub, "features"))
        out_path = f"./predictions/{sub}/ct.nii.gz"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        data = transforms(subject)

        x = data["input"].unsqueeze(0).to(device)
        print(f"Sliding window inference for subject {sub}...")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            pred = sliding_window_inference(
                x, PATCH_SIZE, SW_BATCH, model,
                overlap=OVERLAP, mode="gaussian", progress=True,
                sw_device="cuda", device="cpu",
            )

        # Apply inverse of normalization to get HU
        pred_hu = pred.cpu().numpy()[0, 0] * 3000 - 1000
        affine = data["nacpet"].meta["affine"].numpy()

        print("Saving...")
        nib.save(nib.Nifti1Image(pred_hu, affine), out_path)
        print("Saved:", out_path)


if __name__ == "__main__":

    predict()

from monai.transforms import *


# NOTE: The baseline uses only the NAC-PET as input
# however, your model may use all the images and metadata available
# under the /features folder


def get_transforms(patch_size, num_samples=2):

    transforms = Compose(
        [
            LoadImaged(
                keys=[
                    "nacpet",
                    "mri_combined_in_phase",
                    "mri_combined_out_phase",
                    "ct",
                    "prediction_mask",
                ]
            ),
            EnsureChannelFirstd(
                keys=[
                    "nacpet",
                    "mri_combined_in_phase",
                    "mri_combined_out_phase",
                    "ct",
                    "prediction_mask",
                ]
            ),
            NormalizeIntensityd(
                keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase"],
                nonzero=True,
                channel_wise=True,
                subtrahend=[0],
            ),
            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1000,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            ConcatItemsd(
                keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase"],
                name="input",
            ),
            DeleteItemsd(keys=["nacpet", "mri_combined_in_phase", "mri_combined_out_phase"]),
            # Crop first so all random augmentations run on small patches
            RandSpatialCropSamplesd(
                keys=["input", "ct", "prediction_mask"],
                roi_size=patch_size,
                random_size=False,
                num_samples=num_samples,
            ),
            EnsureTyped(keys=["input", "ct", "prediction_mask"], track_meta=False),
        ]
    )

    return transforms

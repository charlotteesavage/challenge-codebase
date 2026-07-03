from monai.transforms import *


# NOTE: The baseline uses only the NAC-PET as input
# however, your model may use all the images and metadata available
# under the /features folder


class TileTopogramd(MapTransform):
    """Topogram is a 2D frontal projection (X, 1, Z); tile it along the missing
    coronal axis to match the other 3D volumes so it can be concatenated as
    another input channel."""

    def __call__(self, data):
        d = dict(data)
        target_y = d["nacpet"].shape[2]  # channel-first: (C, X, Y, Z)
        for key in self.key_iterator(d):
            d[key] = d[key].repeat(1, 1, target_y, 1)
        return d


def get_transforms(patch_size, num_samples=2):

    transforms = Compose(
        [
            LoadImaged(
                keys=[
                    "nacpet",
                    "topogram",
                    "mri_combined_in_phase",
                    "mri_combined_out_phase",
                    "mri_face_mask",
                    "ct",
                    "prediction_mask",
                ]
            ),
            EnsureChannelFirstd(
                keys=[
                    "nacpet",
                    "topogram",
                    "mri_combined_in_phase",
                    "mri_combined_out_phase",
                    "mri_face_mask",
                    "ct",
                    "prediction_mask",
                ]
            ),
            TileTopogramd(keys=["topogram"]),
            NormalizeIntensityd(
                keys=["nacpet", "topogram", "mri_combined_in_phase", "mri_combined_out_phase", "mri_face_mask"],
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
                keys=["nacpet", "topogram", "mri_combined_in_phase", "mri_combined_out_phase", "mri_face_mask"],
                name="input",
            ),
            DeleteItemsd(keys=["nacpet", "topogram", "mri_combined_in_phase", "mri_combined_out_phase", "mri_face_mask"]),
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

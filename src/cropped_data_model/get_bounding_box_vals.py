import numpy as np
import yaml
from dataset import get_dataset
import nibabel as nib
import os, sys


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def main():

    cfg = load_config()
    data_dir = cfg["data_dir"]
    all_data = get_dataset(data_dir)
    
    # all_data = get_dataset("data")
    subject_ids = []
    centroids = []
    bboxes = []
    for subject in all_data:
        subject_ids.append(os.path.basename(os.path.dirname(os.path.dirname(subject["ct"]))))
        organ = nib.load(subject["organ_seg"]).get_fdata()
        brain = organ == 90
        xs, ys, zs = np.where(brain)
        centroid = (xs.mean(), ys.mean(), zs.mean())
        bbox = ((xs.min(),xs.max()), (ys.min(),ys.max()), (zs.min(),zs.max()))
        centroids.append(centroid)
        bboxes.append(bbox)

    centroids = np.array(centroids)
    centroid_mean = centroids.mean(axis=0)
    centroid_std = centroids.std(axis=0)
    print(f"centroid mean: {centroid_mean}")
    print(f"centroid std: {centroid_std}")

    # flag subjects whose brain centroid sits >3 std from the population mean on any axis --
    # a single bad organ_seg (mislabeled brain) or unusual positioning can otherwise blow up
    # the union bbox below without being noticed
    n_std = 3
    print(f"\nchecking for centroid outliers (> {n_std} std from mean)...")
    outlier_ids = []
    for sub_id, centroid in zip(subject_ids, centroids):
        deviation = np.abs(centroid - centroid_mean) / centroid_std
        flagged_axes = [axis for axis, d in zip("xyz", deviation) if d > n_std]
        if flagged_axes:
            outlier_ids.append(sub_id)
            print(f"  {sub_id}: centroid {centroid} flagged on axis {flagged_axes} (deviation {deviation})")
    if not outlier_ids:
        print("  none found")

    union_min = np.array([min(b[ax][0] for b in bboxes) for ax in range(3)])
    union_max = np.array([max(b[ax][1] for b in bboxes) for ax in range(3)])

    # union bbox min/max is NOT robust to a single subject's bbox -- unlike the centroid
    # (a mean), one stray mislabeled voxel far from the real brain can blow out min/max
    # without moving that subject's centroid enough to get flagged above. So report which
    # subject(s) actually set each extreme, to manually sanity-check before trusting it.
    axis_names = ["x", "y", "z"]
    extreme_report_lines = []
    print("\nunion bbox extremes by subject:")
    for ax, name in enumerate(axis_names):
        min_val = union_min[ax]
        max_val = union_max[ax]
        min_subjects = [sid for sid, b in zip(subject_ids, bboxes) if b[ax][0] == min_val]
        max_subjects = [sid for sid, b in zip(subject_ids, bboxes) if b[ax][1] == max_val]
        line = f"  {name}: min={min_val} contributed by {min_subjects}, max={max_val} contributed by {max_subjects}"
        print(line)
        extreme_report_lines.append(line)

    margin = np.array([25, 25, 20])  # voxels; x/y margin larger since head can rotate/tilt
    vol_shape = np.array(nib.load(all_data[0]["ct"]).shape)

    roi_start = np.clip(union_min - margin, 0, vol_shape)
    roi_end = np.clip(union_max + margin, 0, vol_shape)
    print("\nproposed roi_start:", roi_start)
    print("proposed roi_end:  ", roi_end)
    print("proposed crop size:", roi_end - roi_start)
    script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
    print(script_directory)
    with open(f"{script_directory}/crop_details.txt", "a") as f:
        f.write(f"centroid mean: {centroid_mean}, centroid std: {centroid_std}\n")
        f.write(f"outlier subjects (> {n_std} std): {outlier_ids if outlier_ids else 'none'}\n")
        f.write("union bbox extremes by subject:\n")
        f.write("\n".join(extreme_report_lines) + "\n")
        f.write(f"proposed roi_start: {roi_start}, \nproposed roi_end: {roi_end}, \nproposed crop size: {roi_end - roi_start}\n\n")

if __name__ == "__main__":
    main()



    



# pwd && uv run python3 -c "
# import numpy as np, nibabel as nib
# from dataset import get_dataset
# all_data = get_dataset('../../data')
# for subject in all_data:
#     organ = nib.load(subject['organ_seg']).get_fdata()
#     brain = organ == 90
#     xs, ys, zs = np.where(brain)
#     print(subject['ct'], 'shape', organ.shape)
#     print('  x', xs.min(), xs.max(), ' y', ys.min(), ys.max(), ' z', zs.min(), zs.max())
# "


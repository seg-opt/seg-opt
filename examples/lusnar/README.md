# LuSNAR selected samples

Six selected synchronized frames from the [LuSNAR dataset](https://github.com/zqyu9/LuSNAR-dataset), preserving their original timestamps and scene identifiers in `manifest.json`.

Each timestamp folder contains:

- `rgb.png`: left-camera color image (`image0/color`).
- `label.png`: original segmentation labels (`image0/label`).
- `depth.pfm`: original depth map (`image0/depth`).
- `stereo_rgb.png`: right-camera color image (`image1/color`).
- `lidar.txt`: synchronized LiDAR points (`LiDAR`).

The manifest records each file's original SHA-256 checksum and size. These files are example data, not a training or evaluation split. Consult the upstream dataset for its usage terms and citation requirements.

Sample files use Git LFS. After cloning, run `git lfs install` and `git lfs pull --include="examples/lusnar/**"` to download them.

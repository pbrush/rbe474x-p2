#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_exr(path):
    # 1) Try OpenCV (recommended)
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # float32 HDR
        if img is not None:
            # OpenCV loads color as BGR; convert to RGB if 3-ch
            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[..., :3][:, :, ::-1]
            return img.astype(np.float32)
    except Exception:
        pass

    # 2) Fallback to imageio
    try:
        import imageio.v2 as imageio
        img = imageio.imread(path).astype(np.float32)
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to read EXR with OpenCV and imageio: {e}")

def normalize_for_display(img, p_low=1.0, p_high=99.0):
    """
    Percentile-based normalization, robust to outliers/NaNs/Infs.
    Returns float32 in [0,1].
    """
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if img.ndim == 2:
        # single-channel
        finite = np.isfinite(img)
        vals = img[finite]
        if vals.size == 0:
            return np.zeros_like(img)
        lo, hi = np.percentile(vals, [p_low, p_high])
        if hi <= lo:
            hi = lo + 1e-6
        out = np.clip(img, lo, hi)
        out = (out - lo) / (hi - lo)
        return out

    # multi-channel: compute percentiles on all finite pixels jointly
    finite = np.isfinite(img).all(axis=2)
    vals = img[finite]
    if vals.size == 0:
        return np.zeros_like(img)
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip(img, lo, hi)
    out = (out - lo) / (hi - lo)
    return out

def main():
    ap = argparse.ArgumentParser(description="Visualize EXR (HDR/depth) with robust normalization.")
    ap.add_argument("image_path", type=str, help="Path to .exr")
    ap.add_argument("--p", type=float, nargs=2, default=[1.0, 99.0],
                    help="Low/High percentiles for normalization (default: 1 99)")
    args = ap.parse_args()

    img = read_exr(args.image_path)
    print(f"Loaded: {args.image_path}")
    print(f"Shape: {img.shape}, dtype: {img.dtype}")
    finite = np.isfinite(img)
    if finite.any():
        print(f"Finite min/max: {np.nanmin(img[finite]):.6g} / {np.nanmax(img[finite]):.6g}")
    else:
        print("Warning: no finite values detected; displaying zeros.")

    disp = normalize_for_display(img, p_low=args.p[0], p_high=args.p[1])

    plt.figure()
    if disp.ndim == 2:
        plt.imshow(disp, cmap="viridis")
        plt.colorbar()
    else:
        # If >3 channels, just take first 3
        if disp.shape[2] > 3:
            disp = disp[..., :3]
        plt.imshow(np.clip(disp, 0, 1))
    plt.title("EXR visualization")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()

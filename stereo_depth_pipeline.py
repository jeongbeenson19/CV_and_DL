#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stereo depth pipeline for your own photos.
- Step 1: Calibrate intrinsics with a set of chessboard photos.
- Step 2: Given a stereo pair (two images from the same camera moved between shots), estimate extrinsics via Essential matrix.
- Step 3: If a checkerboard with known square size is visible in both images, compute absolute scale and baseline.
- Step 4: Rectify, compute disparity (SGBM), and reproject to 3D to get a metric depth map.
Author: ChatGPT (GPT-5 Thinking)
"""

import os
import argparse
import glob
import numpy as np
import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_npz(path, **kwargs):
    np.savez(path, **kwargs)


def load_intrinsics(npz_path):
    data = np.load(npz_path)
    K = data['K']
    dist = data['dist']
    img_size = tuple(data['img_size'])
    return K, dist, img_size


def calibrate_intrinsics(calib_dir, pattern_rows, pattern_cols, square_m, out_npz):
    """Calibrate a single camera from multiple chessboard images."""
    # Prepare object points (0,0,0), (1,0,0), ... scaled by square size
    objp = np.zeros((pattern_rows * pattern_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= square_m

    objpoints = []  # 3D points in world space
    imgpoints = []  # 2D points in image plane

    image_paths = sorted(glob.glob(os.path.join(calib_dir, '*.*')))
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {calib_dir}")
    img_size = None

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read {p}")
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])

        ret, corners = cv2.findChessboardCorners(
            img, (pattern_cols, pattern_rows), None)
        if ret:
            # refine corners
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
            corners_ref = cv2.cornerSubPix(
                img, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp.copy())
            imgpoints.append(corners_ref)
        else:
            print(f"[INFO] Chessboard not found in {p}")

    # if len(objpoints) < 5:
    #     raise RuntimeError("At least 5 valid chessboard detections are recommended for reliable calibration.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)
    if not ret:
        print("[WARN] calibrateCamera returned False; proceeding with outputs anyway.")

    # Compute reprojection error
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += err**2
        total_points += len(objpoints[i])
    rms = np.sqrt(total_error / total_points) if total_points > 0 else np.nan

    save_npz(out_npz, K=K, dist=dist, img_size=np.array(img_size), rms=rms)
    print(f"[OK] Intrinsics saved to {out_npz}")
    print(f"     K=\n{K}\n     dist={dist.ravel()}")
    print(f"     Reprojection RMS (px): {rms:.4f}")
    return K, dist, img_size, rms


def match_features(imgL, imgR, max_kp=4000):
    """ORB + BF-Hamming matching with Lowe's ratio and RANSAC filtering later."""
    orb = cv2.ORB_create(nfeatures=max_kp, scaleFactor=1.2,
                         nlevels=8, edgeThreshold=31, WTA_K=2)
    kps1, des1 = orb.detectAndCompute(imgL, None)
    kps2, des2 = orb.detectAndCompute(imgR, None)
    if des1 is None or des2 is None or len(kps1) < 50 or len(kps2) < 50:
        raise RuntimeError(
            "Not enough features detected. Try more texture, better lighting, or increase max_kp.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 50:
        raise RuntimeError("Not enough good matches after ratio test.")
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])
    return pts1, pts2, kps1, kps2, good


def recover_pose_from_essential(K, pts1, pts2, img_size):
    """Estimate E, recover R,t with RANSAC inliers."""
    # Normalize points by intrinsics for findEssentialMat
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")
    inliers = mask.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)
    # t is unit-norm up to scale
    return R, t.reshape(3, 1), inliers


def find_chessboard_corners(img, pattern_rows, pattern_cols, K=None, dist=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ret, corners = cv2.findChessboardCorners(
        gray, (pattern_cols, pattern_rows), None)
    if not ret:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners_ref = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners_ref


def compute_scale_from_checkerboard(imgL, imgR, K, dist, R, t, pattern_rows, pattern_cols, square_m):
    """Triangulate chessboard corners with [I|0] and [R|t] using normalized coords to get scale."""
    ok1, corners1 = find_chessboard_corners(imgL, pattern_rows, pattern_cols)
    ok2, corners2 = find_chessboard_corners(imgR, pattern_rows, pattern_cols)
    if not (ok1 and ok2):
        print("[WARN] Checkerboard not found in both images. Cannot compute absolute scale from board.")
        return None, None

    # Undistort to normalized coordinates
    pts1n = cv2.undistortPoints(corners1, K, dist)
    pts2n = cv2.undistortPoints(corners2, K, dist)
    # Build projection matrices for normalized coords
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t])

    # Reshape to 2xN
    p1 = pts1n.reshape(-1, 2).T
    p2 = pts2n.reshape(-1, 2).T

    X_h = cv2.triangulatePoints(P1, P2, p1, p2)
    X = (X_h[:3] / X_h[3]).T  # Nx3 in relative units

    # Compute mean edge length (horizontal and vertical neighbors) in reconstructed 3D
    rows, cols = pattern_rows, pattern_cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if c + 1 < cols:
                idx_right = r * cols + (c + 1)
                edges.append(np.linalg.norm(X[idx] - X[idx_right]))
            if r + 1 < rows:
                idx_down = (r + 1) * cols + c
                edges.append(np.linalg.norm(X[idx] - X[idx_down]))
    edges = np.array(edges, dtype=np.float64)
    mean_edge = float(np.mean(edges))
    if mean_edge <= 0 or not np.isfinite(mean_edge):
        print("[WARN] Invalid reconstructed edge length; cannot scale.")
        return None, None

    scale = square_m / mean_edge  # meters per relative-unit
    baseline_rel = float(np.linalg.norm(t))
    baseline_m = scale * baseline_rel
    return scale, baseline_m


def stereo_rectify_and_disparity(imgL, imgR, K, dist, R, t, out_dir, min_disp=0, num_disp=160, block_size=5):
    """Rectify and compute disparity with SGBM. Returns disparity (float32, px) and Q."""
    h, w = imgL.shape[:2]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    map1x, map1y = cv2.initUndistortRectifyMap(
        K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # SGBM parameters tuned conservatively
    num_disp = int(np.ceil(num_disp / 16.0)) * 16  # must be divisible by 16
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    grayL = cv2.cvtColor(
        rectL, cv2.COLOR_BGR2GRAY) if rectL.ndim == 3 else rectL
    grayR = cv2.cvtColor(
        rectR, cv2.COLOR_BGR2GRAY) if rectR.ndim == 3 else rectR
    disp = sgbm.compute(grayL, grayR).astype(
        np.float32) / 16.0  # to float pixels

    # Save rectified pair and raw disparity viz
    ensure_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir, 'rect_left.png'), rectL)
    cv2.imwrite(os.path.join(out_dir, 'rect_right.png'), rectR)
    # Visualization with colormap (clip for display)
    disp_vis = disp.copy()
    valid = disp_vis > 0
    if np.any(valid):
        vmin = np.percentile(disp_vis[valid], 2.0)
        vmax = np.percentile(disp_vis[valid], 98.0)
        disp_vis = np.clip((disp_vis - vmin) / (vmax - vmin + 1e-9), 0, 1)
        disp_vis = (disp_vis * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(out_dir, 'disparity_vis.png'), disp_color)
    else:
        print(
            "[WARN] No positive disparities found; check baseline direction or parameters.")

    return disp, Q, rectL, rectR


def depth_from_Q(disparity, Q):
    """Reproject disparity to 3D and extract Z (depth in the rectified left camera frame)."""
    points_3d = cv2.reprojectImageTo3D(
        disparity, Q, handleMissingValues=True)  # shape (H,W,3)
    Z = points_3d[:, :, 2]
    return Z, points_3d


def save_depth(out_dir, depth_m):
    """Save depth as .npy (float32 meters) and as a clipped visualization PNG."""
    ensure_dir(out_dir)
    path_npy = os.path.join(out_dir, 'depth_m.npy')
    np.save(path_npy, depth_m.astype(np.float32))

    # Visualization (clip to robust range, e.g., 1st-99th percentile of valid values)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if np.any(valid):
        low = np.percentile(depth_m[valid], 1.0)
        high = np.percentile(depth_m[valid], 99.0)
        depth_vis = np.clip((depth_m - low) / (high - low + 1e-9), 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(255 - depth_vis, cv2.COLORMAP_PLASMA)
        cv2.imwrite(os.path.join(out_dir, 'depth_vis.png'), depth_color)
    else:
        print("[WARN] No valid positive depths for visualization.")

    return path_npy, os.path.join(out_dir, 'depth_vis.png')


def main():
    parser = argparse.ArgumentParser(
        description="Stereo depth from two images with checkerboard-based metric scaling.")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # Subcommand: calibrate intrinsics
    p_cal = sub.add_parser(
        'calibrate', help='Calibrate camera intrinsics from a folder of chessboard images.')
    p_cal.add_argument('--calib_dir', required=True,
                       help='Folder containing chessboard images.')
    p_cal.add_argument('--pattern_rows', type=int, required=True,
                       help='Number of inner corners per chessboard row.')
    p_cal.add_argument('--pattern_cols', type=int, required=True,
                       help='Number of inner corners per chessboard column.')
    p_cal.add_argument('--square_m', type=float, required=True,
                       help='Square size in meters (e.g., 0.0245).')
    p_cal.add_argument('--out', default='intrinsics.npz',
                       help='Output npz path.')

    # Subcommand: depth from a pair
    p_depth = sub.add_parser(
        'depth', help='Estimate depth from a stereo pair.')
    p_depth.add_argument('--left', required=True,
                         help='Left image (first view).')
    p_depth.add_argument('--right', required=True,
                         help='Right image (second view).')
    p_depth.add_argument('--intrinsics', default='intrinsics.npz',
                         help='Path to intrinsics npz (from calibrate).')
    p_depth.add_argument('--pattern_rows', type=int, required=True,
                         help='Chessboard inner rows (for scaling).')
    p_depth.add_argument('--pattern_cols', type=int, required=True,
                         help='Chessboard inner cols (for scaling).')
    p_depth.add_argument('--square_m', type=float,
                         required=True, help='Square size in meters.')
    p_depth.add_argument('--baseline_m', type=float, default=None,
                         help='Optional known baseline (meters). If set, overrides checkerboard scaling.')
    p_depth.add_argument('--out_dir', default='out', help='Output directory.')
    p_depth.add_argument('--num_disp', type=int, default=160,
                         help='SGBM numDisparities (multiple of 16).')
    p_depth.add_argument('--block_size', type=int, default=5,
                         help='SGBM block size (odd, 3..11).')

    args = parser.parse_args()

    if args.cmd == 'calibrate':
        K, dist, img_size, rms = calibrate_intrinsics(
            args.calib_dir, args.pattern_rows, args.pattern_cols, args.square_m, args.out)
        return

    if args.cmd == 'depth':
        # Load images
        imgL = cv2.imread(args.left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(args.right, cv2.IMREAD_COLOR)
        if imgL is None or imgR is None:
            raise FileNotFoundError("Could not read left or right image.")
        # Load intrinsics
        K, dist, img_size = load_intrinsics(args.intrinsics)
        # Feature matching & pose
        pts1, pts2, kps1, kps2, matches = match_features(imgL, imgR)
        R, t_rel, inliers = recover_pose_from_essential(
            K, pts1, pts2, imgL.shape[:2][::-1])
        print(
            f"[INFO] Relative translation (unit-norm): t_rel = {t_rel.ravel()}  (||t||={np.linalg.norm(t_rel):.4f})")

        # Determine scale & baseline
        scale = None
        baseline_m = None
        if args.baseline_m is not None:
            baseline_m = float(args.baseline_m)
            scale = baseline_m / float(np.linalg.norm(t_rel))
            print(
                f"[OK] Using provided baseline_m = {baseline_m:.6f} m -> scale = {scale:.6f}")
        else:
            scale, baseline_m = compute_scale_from_checkerboard(
                imgL, imgR, K, dist, R, t_rel, args.pattern_rows, args.pattern_cols, args.square_m)
            if scale is not None:
                print(
                    f"[OK] Scale from checkerboard = {scale:.6f} (m/rel-unit); baseline ≈ {baseline_m:.6f} m")
            else:
                print(
                    "[WARN] No absolute scale available; results will be up-to-scale only.")

        # Use scaled translation for rectification (improves Q matrix metric correctness)
        t_for_rect = t_rel.copy()
        if scale is not None:
            t_for_rect *= scale

        # Rectify & disparity
        disp, Q, rectL, rectR = stereo_rectify_and_disparity(imgL, imgR, K, dist, R, t_for_rect, args.out_dir,
                                                             num_disp=args.num_disp, block_size=args.block_size)

        # Depth
        depth_m, points_3d = depth_from_Q(disp, Q)

        # Save outputs
        ensure_dir(args.out_dir)
        # Save disparity raw (float32)
        np.save(os.path.join(args.out_dir, 'disparity_px.npy'),
                disp.astype(np.float32))
        path_depth_npy, path_depth_vis = save_depth(args.out_dir, depth_m)

        # Report
        if baseline_m is not None and np.isfinite(baseline_m):
            print(f"[RESULT] Estimated baseline: {baseline_m:.6f} m")
        else:
            print("[RESULT] Baseline unknown (relative only). Depth is not metric.")

        # Optional sanity check: estimate mean chessboard edge from 3D reprojected points (if we have corners)
        ok1, c1 = find_chessboard_corners(
            rectL, args.pattern_rows, args.pattern_cols)
        ok2, c2 = find_chessboard_corners(
            rectR, args.pattern_rows, args.pattern_cols)
        if ok1 and ok2:
            # Disparity at corner locations -> 3D points from Q
            cn1 = c1.reshape(-1, 2)
            disp_vals = []
            xyz = []
            H, W = disp.shape
            for pt in cn1:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < W and 0 <= y < H:
                    d = disp[y, x]
                    if d > 0 and np.isfinite(d):
                        X = points_3d[y, x, :]
                        xyz.append(X)
                        disp_vals.append(d)
            xyz = np.array(xyz, dtype=np.float64)
            if xyz.shape[0] >= 4:
                # compute edge mean in metric (should be ~square_m)
                rows, cols = args.pattern_rows, args.pattern_cols
                # Note: corner order may differ due to rectification; skip rigorous indexing check
                # Here we approximate by taking nearest-neighbor edges in the grid order.
                # This is a loose sanity check, not a strict evaluation.
                mean_edge = 0.0
                cnt = 0
                grid = xyz.reshape(
                    rows*cols, 3) if xyz.shape[0] == rows*cols else xyz[:rows*cols]
                for r in range(rows):
                    for c in range(cols):
                        idx = r * cols + c
                        if (c + 1) < cols:
                            mean_edge += np.linalg.norm(
                                grid[idx] - grid[idx+1])
                            cnt += 1
                        if (r + 1) < rows:
                            mean_edge += np.linalg.norm(
                                grid[idx] - grid[idx+cols])
                            cnt += 1
                if cnt > 0:
                    mean_edge /= cnt
                    print(
                        f"[CHECK] Mean reconstructed chessboard edge ≈ {mean_edge:.6f} m (gt square_m={args.square_m:.6f} m)")
        else:
            print(
                "[INFO] Chessboard not reliably detected in rectified images for sanity check.")

        print(f"[OK] Saved:")
        print(
            f"  - Rectified images: {os.path.join(args.out_dir, 'rect_left.png')}, rect_right.png")
        print(
            f"  - Disparity: {os.path.join(args.out_dir, 'disparity_px.npy')} (float32 pixels), 'disparity_vis.png' for preview")
        print(
            f"  - Depth: {path_depth_npy} (float32 meters), 'depth_vis.png' for preview")


if __name__ == '__main__':
    main()

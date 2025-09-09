import cv2
import numpy as np
import glob
import os

# ==== 설정 ====
CHECKERBOARD = (18, 13)  # (가로 코너 수, 세로 코너 수)
USE_CLAHE = True         # True: CLAHE, False: 전역 equalizeHist
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
# ==============

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

objpoints = []  # 3D 점
imgpoints = []  # 2D 점

# 3D 체커보드 좌표 준비
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)

images = sorted(glob.glob('./cam_calibration/*.jpg'))
cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)

# CLAHE 핸들 준비(옵션)
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                        tileGridSize=CLAHE_TILE) if USE_CLAHE else None

for i, fname in enumerate(images, 1):
    img = cv2.imread(fname)
    if img is None:
        print(f"[{i}] {fname} → 읽기 실패")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------- 히스토그램 평활화 --------
    if USE_CLAHE:
        gray_eq = clahe.apply(gray)
    else:
        gray_eq = cv2.equalizeHist(gray)
    # ----------------------------------

    # (옵션) 미세한 노이즈 완화가 필요하면 주석 해제
    # gray_eq = cv2.GaussianBlur(gray_eq, (3, 3), 0)

    # SB(더 견고한 체스보드 탐지기) 사용
    ret, corners = cv2.findChessboardCornersSB(
        gray_eq,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    print(f"[{i}/{len(images)}] {os.path.basename(fname)} → ret={ret}, "
          f"equalize={'CLAHE' if USE_CLAHE else 'global'}")

    vis = img.copy()

    if ret:
        # cornerSubPix 보정(정밀화)
        corners2 = cv2.cornerSubPix(
            gray_eq, corners, (11, 11), (-1, -1), criteria
        )
        objpoints.append(objp)
        imgpoints.append(corners2)
        vis = cv2.drawChessboardCorners(vis, CHECKERBOARD, corners2, True)
        print(
            f"   코너 개수: {len(corners2)} (기대값: {CHECKERBOARD[0]*CHECKERBOARD[1]})")
    else:
        print("   ❌ 코너 탐지 실패")

    # 보기 편하도록 너무 큰 이미지는 리사이즈(최대 변 1000px)
    h, w = vis.shape[:2]
    scale = 1000 / max(h, w)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(w*scale), int(h*scale)),
                         interpolation=cv2.INTER_AREA)

    cv2.imshow('img', vis)
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC → 종료
        break

cv2.destroyAllWindows()

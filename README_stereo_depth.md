
# Stereo Depth Pipeline (Single Camera, Two Shots)

## 개요
- **Intrinsics(내부 파라미터)**: 체커보드 여러 장으로 보정.
- **Extrinsics(외부 파라미터)**: 동일 카메라로 촬영한 두 장에서 특징점 매칭 → Essential Matrix → `R, t` 복원(스케일 미정).
- **스케일(절대 길이)**: 두 이미지에 **체커보드가 함께 보이면**, 코너 3D 재구성으로 평균 변 길이를 **정확한 사각형 한 변 길이(`square_m`)**에 맞춰 절대 스케일 복원.
- **Rectify + SGBM**: 정렬/보정 후 SGBM으로 disparity 추정, `cv2.reprojectImageTo3D`로 **미터 단위 깊이맵** 생성.

## 준비물
1. **체커보드 규격**: (pattern_rows, pattern_cols) = 내부 코너 개수. 예: 7×10.
2. **한 칸 변 길이(square_m)**: 예: 0.0245 m.
3. **캘리브레이션 이미지**: 같은 카메라로 찍은 체커보드 여러 장.
4. **스테레오 쌍**: 동일 장면을 baseline(가까움/중간/멀리)을 달리해 두 장 촬영. (가능하면 체커보드 포함)

## 설치
```bash
pip install opencv-python numpy
```

## 사용법
### 1) Intrinsics 보정
```bash
python stereo_depth_pipeline.py calibrate \
  --calib_dir data/calib \
  --pattern_rows 7 --pattern_cols 10 \
  --square_m 0.0245 \
  --out intrinsics.npz
```

### 2) 깊이 추정
- **체커보드로 스케일 자동 복원**:
```bash
python stereo_depth_pipeline.py depth \
  --left data/pairs/near_L.jpg --right data/pairs/near_R.jpg \
  --intrinsics intrinsics.npz \
  --pattern_rows 7 --pattern_cols 10 \
  --square_m 0.0245 \
  --out_dir out/near
```
- **체커보드가 없고 baseline을 직접 측정한 경우**(예: 0.12 m):
```bash
python stereo_depth_pipeline.py depth \
  --left data/pairs/mid_L.jpg --right data/pairs/mid_R.jpg \
  --intrinsics intrinsics.npz \
  --pattern_rows 7 --pattern_cols 10 \
  --square_m 0.0245 \
  --baseline_m 0.12 \
  --out_dir out/mid
```

## 팁
- 촬영 시 **수평 이동(baseline)** 위주로 움직이고, 회전은 최소화.
- **같은 노출/ISO**로 촬영하면 매칭 품질↑.
- SGBM 파라미터(`--num_disp`, `--block_size`)는 장면/해상도에 맞춰 조절.
- 결과 폴더에는 `rect_left/right.png`, `disparity_vis.png`, `disparity_px.npy`, `depth_vis.png`, `depth_m.npy`가 저장됩니다.

## 베이스라인 실험
- `out/near`, `out/mid`, `out/far`로 각각 실행해 **baseline 추정치**와 깊이 퀄리티를 비교하세요.
- 콘솔 로그의 `[RESULT] Estimated baseline`과 `[CHECK] Mean reconstructed chessboard edge`를 확인.

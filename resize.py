import cv2
import os


def resize_image(input_path, output_path, width=None, height=None, scale=None):
    """
    이미지를 원하는 크기(width, height)나 비율(scale)로 리사이즈합니다.
    - width, height 중 하나만 주면 비율을 유지하며 리사이즈
    - scale을 주면 단순히 배율로 리사이즈
    """

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {input_path}")

    h, w = img.shape[:2]

    # 비율 기반 리사이즈
    if scale is not None:
        new_w, new_h = int(w * scale), int(h * scale)
    # width, height 기반 리사이즈
    elif width is not None and height is None:
        scale = width / w
        new_w, new_h = width, int(h * scale)
    elif height is not None and width is None:
        scale = height / h
        new_w, new_h = int(w * scale), height
    elif width is not None and height is not None:
        new_w, new_h = width, height
    else:
        raise ValueError("width/height 또는 scale 중 하나를 지정해야 합니다.")

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_path, resized)
    print(f"[OK] Saved resized image to {output_path} (size={new_w}x{new_h})")


if __name__ == "__main__":
    # 사용 예시
    in_path = "/Users/mac_air/CV_and_DL/pairs/IMG_3462.jpeg"
    out_path = "/Users/mac_air/CV_and_DL/pairs/IMG_3462_resized.jpeg"

    # 1) 특정 width 기준 (가로 1280으로 고정, 세로는 자동 비율)
    resize_image(in_path, out_path, width=512)

    # 2) 특정 height 기준 (세로 720으로 고정)
    # resize_image(in_path, out_path, height=720)

    # 3) 비율 기준 (절반 크기)
    # resize_image(in_path, out_path, scale=0.5)

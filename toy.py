import cv2

# 사람 검출기 초기화
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
threshold = 0.6

# 비디오 열기
cap = cv2.VideoCapture("sample2.mp4")
paused = False
frame = None
ellipses = []  # [(center, axes)]
selected_points = []
lines = []

# 마우스 콜백 함수


def click_event(event, x, y, flags, param):
    global selected_points, lines, ellipses

    margin = 50  # 중심 주변 10픽셀 여유 추가

    if event == cv2.EVENT_LBUTTONDOWN:
        for center, axes in ellipses:
            cx, cy = center
            ax, ay = axes
            # 타원 내부 판단 + margin 적용
            if ((x - cx)**2) / ((ax + margin)**2) + ((y - cy)**2) / ((ay + margin)**2) <= 1:
                cv2.ellipse(display, center, axes, 0, 0, 360, (0, 0, 0), -1)
                selected_points.append(center)
                break

        if len(selected_points) == 2:
            pt1, pt2 = selected_points
            lines.append((pt1, pt2))
            selected_points = []

    elif event == cv2.EVENT_RBUTTONDOWN:
        lines.clear()
        selected_points.clear()


# 윈도우 설정 및 콜백 등록
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", click_event)

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("영상이 끝났거나 오류 발생")
            break

    display = frame.copy()

    # 타원 다시 그림
    for center, axes in ellipses:
        cv2.ellipse(display, center, axes, 0, 0, 360, (50, 50, 50), -1)

    # 선 다시 그림
    for pt1, pt2 in lines:
        cv2.line(display, pt1, pt2, (0, 0, 255), 5)

    # 출력
    cv2.imshow("Video", display)
    key = cv2.waitKey(30) & 0xFF

    if key == 27:  # ESC
        break

    elif key == ord(' '):  # 스페이스바
        paused = not paused
        if paused:
            # 사람 검출 수행
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            eq = clahe.apply(gray)
            color_eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

            boxes, weights = hog.detectMultiScale(
                color_eq, winStride=(4, 4), padding=(8, 8), scale=1.03
            )

            ellipses.clear()
            for (box, weight) in zip(boxes, weights):
                if weight < threshold:
                    continue
                x, y, w, h = box
                center = (x + w // 2, y + h)
                axes = (w // 3, h // 15)
                ellipses.append((center, axes))

cap.release()
cv2.destroyAllWindows()

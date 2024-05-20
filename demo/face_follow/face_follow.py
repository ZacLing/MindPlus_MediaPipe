import cv2
import mediapipe as mp
from units import MultiDimensionalPIDController, SWG_filter
from pinpong.libs.microbit_motor import DFServo
from pinpong.board import Board
import numpy as np

DISTANCE = 30
Board().begin()

horizontal_servo = DFServo(8)
vertical_servo = DFServo(7)
current_h = 90
current_v = 90
horizontal_servo.angle(current_h)
vertical_servo.angle(current_v)

# 0.05, 0.07
pid = MultiDimensionalPIDController(Kp=1, Ki=0, Kd=0)
swg = SWG_filter(window_size=1)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cv2.namedWindow('MediaPipe Face Detection', cv2.WND_PROP_FULLSCREEN)    # Set the windows to be full screen.
cv2.setWindowProperty('MediaPipe Face Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    # Set the windows to be full screen.

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        mp_drawing.draw_detection(image, detection)
        # 提取检测框的坐标
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x_center = int(bboxC.xmin * iw + (bboxC.width * iw) / 2)
        y_center = int(bboxC.ymin * ih + (bboxC.height * ih) / 2)
        # 将中心点坐标转换为以全屏幕中心为原点的坐标系
        x_center_new = x_center - iw // 2
        y_center_new = ih // 2 - y_center
        now_point = swg.compute([x_center_new, y_center_new])
        set_point = [0.0, 0.0]
        error = pid.compute(now_point, set_point)


        # 计算水平和垂直方向的转角
        alpha_h = np.arctan(error[0] * 0.5 / DISTANCE) * (180 / np.pi) * 0.1
        alpha_v = np.arctan(error[1] * 0.5 / DISTANCE) * (180 / np.pi) * 0.1
        print(f"alpha_h: {alpha_h}, alpha_v: {alpha_v}")

        # 更新伺服电机的角度
        current_h -= alpha_h
        current_v -= alpha_v

        # 角度限制在0到180度之间
        current_h = int(max(90-50, min(90+50, current_h)))
        current_v = int(max(90-50, min(90+50, current_v)))

        horizontal_servo.angle(current_h)
        vertical_servo.angle(current_v)

        print(f"Horizontal Angle: {current_h}, Vertical Angle: {current_v}")

        # 在图像上绘制中心点
        cv2.circle(image, (x_center, y_center), 5, (0, 255, 0), -1)
        # 显示中心点坐标
        cv2.putText(image, f'Center: ({x_center_new}, {y_center_new})', (x_center, y_center - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Flip the image horizontally for a selfie-view display.
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.flip(image, 1)
    cv2.imshow('MediaPipe Face Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


import cv2 as cv
import  mediapipe as mp
import numpy as np

import time
import  matplotlib.pyplot as plt

# 定义可视化图像函数
def look_img(img):
    img_RGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

# 导入三维人脸关键点检测模型
mp_face_mesh=mp.solutions.face_mesh
# help(mp_face_mesh.FaceMesh)

model=mp_face_mesh.FaceMesh(
    static_image_mode=False,#TRUE:静态图片/False:摄像头实时读取
    
    max_num_faces=5,
    min_detection_confidence=0.5, #置信度阈值，越接近1越准
    min_tracking_confidence=0.5,#追踪阈值
)


# 导入可视化函数和可视化样式
mp_drawing=mp.solutions.drawing_utils
# mp_drawing_styles=mp.solutions.drawing_styles
draw_spec=mp_drawing.DrawingSpec(thickness=2,circle_radius=1,color=[66,77,229])
landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=2,color=[66,77,229])
# 轮廓可视化
connection_drawing_spec=mp_drawing.DrawingSpec(thickness=2,circle_radius=1,color=[233,155,6])



# 处理帧函数
def process_frame(img):
    start_time = time.time()
    scaler = 1
    h, w = img.shape[0], img.shape[1]
    img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = model.process(img_RGB)
    if results.multi_face_landmarks:
        # for face_landmarks in results.multi_face_landmarks:
            # 连轮廓最左侧点
            FL = results.multi_face_landmarks[0].landmark[234];
            FL_X, FL_Y = int(FL.x * w), int(FL.y * h);
            FL_Color = (234, 0, 255)
            img = cv.circle(img, (FL_X, FL_Y), 5, FL_Color, -1)

            # 左眼左眼角
            ELL = results.multi_face_landmarks[0].landmark[33];  # 33坐标为上图中标注的点的序号
            ELL_X, ELL_Y = int(ELL.x * w), int(ELL.y * h);
            ELL_Color = (0, 255, 0)
            img = cv.circle(img, (ELL_X, ELL_Y), 5, ELL_Color, -1)
           

            # 左眼右眼角
            ELR = results.multi_face_landmarks[0].landmark[133];  # 133坐标为上图中标注的点的序号
            ELR_X, ELR_Y = int(ELR.x * w), int(ELR.y * h);
            ELR_Color = (0, 255, 0)
            img = cv.circle(img, (ELR_X, ELR_Y), 5, ELR_Color, -1)
          

            # 右眼左眼角362
            ERL = results.multi_face_landmarks[0].landmark[362];  # 133坐标为上图中标注的点的序号
            ERL_X, ERL_Y = int(ERL.x * w), int(ERL.y * h);
            ERL_Color = (233, 255, 128)
            img = cv.circle(img, (ERL_X, ERL_Y), 5, ERL_Color, -1)
           

            # 右眼右眼角263
            ERR = results.multi_face_landmarks[0].landmark[263];  # 133坐标为上图中标注的点的序号
            ERR_X, ERR_Y = int(ERR.x * w), int(ERR.y * h);
            ERR_Color = (23, 255, 128)
            img = cv.circle(img, (ERR_X, ERR_Y), 5, ERR_Color, -1)
          
            # 轮廓最右侧
            FR = results.multi_face_landmarks[0].landmark[454];  # 454 坐标为上图中标注的点的序号
            FR_X, FR_Y = int(FR.x * w), int(FR.y * h);
            FR_Color = (0, 255, 0)
            img = cv.circle(img, (FR_X, FR_Y), 5, FR_Color, -1)
                       
            # 脸上侧边缘
            FT = results.multi_face_landmarks[0].landmark[10];  # 10 坐标为上图中标注的点的序号
            FT_X, FT_Y = int(FT.x * w), int(FT.y * h);
            FT_Color = (231, 141, 181)
            img = cv.circle(img, (FT_X, FT_Y), 5, FT_Color, -1)
 
            # 脸下侧边缘
            FB = results.multi_face_landmarks[0].landmark[152];  # 152 坐标为上图中标注的点的序号
            FB_X, FB_Y = int(FB.x * w), int(FB.y * h);
            FB_Color = (231, 141, 181)
            img = cv.circle(img, (FB_X, FB_Y), 5, FB_Color, -1)
            
            # 从左往右六个点的横坐标
            Six_X = np.array([FL_X, ELL_X, ELR_X, ERL_X, ERR_X, FR_X])

            # 从最左到最右的距离
            Left_Right = FR_X - FL_X
            # 从左向右六个点的间隔的五个距离一并划归
            Five_Distance = 100 * np.diff(Six_X) / Left_Right

            # 两眼宽度的平均值
            Eye_Width_Mean = np.mean((Five_Distance[1], Five_Distance[3]))

            # 五个距离分别与两眼宽度均值的差
            Five_Eye_Diff = Five_Distance - Eye_Width_Mean

            # 求L2范数，作为颜值的指标
            Five_Eye_Metrics = np.linalg.norm(Five_Eye_Diff)

            cv.line(img, (FL_X, FT_Y), (FL_X, FB_Y), FL_Color, 3)
            cv.line(img, (ELL_X, FT_Y), (ELL_X, FB_Y), ELL_Color, 3)
            cv.line(img, (ELR_X, FT_Y), (ELR_X, FB_Y), ELR_Color, 3)
            cv.line(img, (ERL_X, FT_Y), (ERL_X, FB_Y), ERL_Color, 3)
            cv.line(img, (ERR_X, FT_Y), (ERR_X, FB_Y), ERR_Color, 3)
            cv.line(img, (FR_X, FT_Y), (FR_X, FB_Y), FR_Color, 3)
            cv.line(img, (FL_X, FT_Y), (FR_X, FT_Y), FT_Color, 3)
            cv.line(img, (FL_X, FB_Y), (FR_X, FB_Y), FB_Color, 3)

            scaler = 1
            #五眼指标
            img = cv.putText(img, 'Five Eye Metrics{:.2f}'.format(Five_Eye_Metrics), (25, 50), cv.FONT_HERSHEY_SIMPLEX,
                             1,
                             (218, 112, 214), 2, 6)
            img = cv.putText(img, 'Distance 1{:.2f}'.format(Five_Eye_Diff[0]), (25, 100), cv.FONT_HERSHEY_SIMPLEX, 1,
                             (218, 112, 214), 2, 5)
            img = cv.putText(img, 'Distance 1{:.2f}'.format(Five_Eye_Diff[2]), (25, 150), cv.FONT_HERSHEY_SIMPLEX, 1,
                             (218, 112, 214), 2, 4)
            img = cv.putText(img, 'Distance 1{:.2f}'.format(Five_Eye_Diff[4]), (25, 200), cv.FONT_HERSHEY_SIMPLEX, 1,
                             (218, 112, 214), 2, 4)


    else:
        img = cv.putText(img, 'NO FACE DELECTED', (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1.25,
                         (218, 112, 214), 1, 8)

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像的帧数FPS
    FPS = 1 / (end_time - start_time)
    scaler = 1
    img = cv.putText(img, 'FPS' + str(int(FPS)), (25 * scaler, 300 * scaler), cv.FONT_HERSHEY_SIMPLEX,
                         1.25 * scaler, (0, 0, 255), 1, 8)
    return img

# 调用摄像头
cap=cv.VideoCapture(0)


cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
cv.namedWindow('my_window',cv.WND_PROP_FULLSCREEN)    #Set the windows to be full screen.
cv.setWindowProperty('my_window', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)    #Set the windows to be full screen.

cap.open(0)
# 无限循环，直到break被触发
while cap.isOpened():
    success,frame=cap.read()
    # if not success:
    #     print('ERROR')
    #     break
    frame=process_frame(frame)
    #展示处理后的三通道图像
    cv.imshow('my_window',frame)
    if cv.waitKey(1) &0xff==ord('q'):
        break

cap.release()
cv.destroyAllWindows()



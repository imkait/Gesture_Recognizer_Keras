#https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=zh-tw

import os
os.environ["KERAS_BACKEND"] = "torch"

import sys
import time
import numpy as np
import cv2
import mediapipe as mp
import keras

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 計算 FPS 的全域變數
COUNTER, FPS = 0, 0
START_TIME = time.time()
DETECTION_RESULT = None

# 計算手指各點距離函數
def compute_distances(landmarks):
    distances = []
    # 定義各點到手掌根部，以及手指兩兩指尖配對
    pairs = [(0, 1), (0, 2), (0, 3), (0, 4),
            (0, 5), (0, 6), (0, 7), (0, 8),
            (0, 9), (0, 10), (0, 11), (0, 12),
            (0, 13), (0, 14), (0, 15), (0, 16),
            (0, 17), (0, 18), (0, 19), (0, 20),
            (4, 8), (8, 12), (12, 16), (16, 20)]

    # 以座標0和座標9作為基準點，計算基準距離
    ref_pair = (0, 9)
    rx1,ry1,rz1 = landmarks[ref_pair[0]].x, landmarks[ref_pair[0]].y, landmarks[ref_pair[0]].z
    rx2,ry2,rz2 = landmarks[ref_pair[1]].x, landmarks[ref_pair[1]].y, landmarks[ref_pair[1]].z
    r_dis=( (rx1-rx2)**2+(ry1-ry2)**2+(rz1-rz2)**2 )**0.5
    

    #計算定義列表中各點距離，並新增到distances串列中
    for pair in pairs:
        x1,y1,z1 = landmarks[pair[0]].x, landmarks[pair[0]].y, landmarks[pair[0]].z
        x2,y2,z2 = landmarks[pair[1]].x, landmarks[pair[1]].y, landmarks[pair[1]].z
        dis=( (x1-x2)**2+(y1-y2)**2+(z1-z2)**2 )**0.5
        distance = dis / r_dis
        distances.append(distance)

    return distances


# 載入DNN模型
model_filename = "dnn_model.keras"
dnn_model = keras.saving.load_model(model_filename)


# 載入標籤
label_file = "labels.txt"
with open(label_file, 'r') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

# 主程式
def run(model: str, num_hands: int,
        min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float,
        camera_id: int, width: int, height: int) -> None:


  # 設定影像擷取畫面的寬高
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


  # FPS文字參數
  row_size = 50  # pixels
  left_margin = 20  # pixels
  text_color = (0, 0, 0)  # black
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10


  def save_result(result: vision.HandLandmarkerResult,
                  unused_output_image: mp.Image, timestamp_ms: int):
      global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # 計算FPS
      if COUNTER % fps_avg_frame_count == 0:
          FPS = fps_avg_frame_count / (time.time() - START_TIME)
          START_TIME = time.time()
      # 指定推論結果為DETECTION_RESULT
      DETECTION_RESULT = result
      COUNTER += 1

  # 初始化手部地標偵測模型
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.HandLandmarkerOptions(
      base_options=base_options,
      running_mode=vision.RunningMode.LIVE_STREAM,
      num_hands=num_hands,
      min_hand_detection_confidence=min_hand_detection_confidence,
      min_hand_presence_confidence=min_hand_presence_confidence,
      min_tracking_confidence=min_tracking_confidence,
      result_callback=save_result)
  detector = vision.HandLandmarker.create_from_options(options)
  
  # 當鏡頭開啟時
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit('錯誤: 無法讀取webcam. 請確認你的webcam設定。')

    # 翻轉畫面
    image = cv2.flip(image, 1)

    # 轉換影像依據模型需要，從BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 使用模型對影像檢測
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    # 顯示 FPS
    fps_text = 'FPS = {:.1f}'.format(FPS)
    text_location = (left_margin, row_size)
    current_frame = image
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                font_size, text_color, font_thickness, cv2.LINE_AA)

    
    if DETECTION_RESULT:
      for idx in range(len(DETECTION_RESULT.hand_landmarks)):
        hand_landmarks = DETECTION_RESULT.hand_landmarks[idx]                
        distances = compute_distances(hand_landmarks)
        #print(distances)
        
        # 將格式轉成1筆資料24個特徵(1,24)
        hand_dis=np.array(distances).reshape(1,24)
        #print(hand_dis)
        #print(hand_dis.shape)

        # 將資料送到模型預測
        prediction = dnn_model.predict(hand_dis)

        # 檢視預測結果
        #print(prediction)

        # 取得最高分數的索引
        yid=np.argmax(prediction)

        # 取得分數
        np_score=np.max(prediction)
        # 將numpy數值轉成float16
        score=np_score.astype(np.float16)
        # 如果分數大於0.5，才顯示文字
        if score > 0.5:
          dis_txt=f"Hand: {labels[yid]}  Score: {round(score,2)} "          
          cv2.putText(current_frame, dis_txt, (left_margin, row_size + 50 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)         

        #將座標點轉換格式加入hand_landmarks_proto陣列
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                          z=landmark.z) for landmark in hand_landmarks
        ])
        # 畫出手部的點與線
        mp_drawing.draw_landmarks(
          current_frame,
          hand_landmarks_proto,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
        
    cv2.imshow('hand_landmarker', current_frame)


    # ESC鍵離開程式
    if cv2.waitKey(1) == 27:
      break
  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  run('hand_landmarker.task', 1, 0.5 , 0.5 , 0.5 , 0 , 1280, 720)
  
  #參數說明:
  # 模型檔名: 'hand_landmarker.task'
  # 手的數量: 1
  # 最低手部偵測信心分數: 0.5 
  # 最低手部存在信心分數: 0.5 
  # 最低手部追蹤成功信心分數:0.5
  # 相機編號: 0
  # 畫面寬度: 1280
  # 畫面高度: 720
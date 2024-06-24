import os
os.environ["KERAS_BACKEND"] = "torch"

#Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import keras
import numpy as np
from sklearn.model_selection import train_test_split

# 設定檔案路徑
dataset_path='dataset/'

# 將檔案讀入files串列
files = [f for f in os.listdir(dataset_path) if f.endswith('.npy')]

# x是訓練資料特徵值
x = []
# y是實際值
y = []

labels=[]

# 以迭代的方式載入files
for idx,file in enumerate(files):
    data = np.load(os.path.join(dataset_path, file))
    # 將資料擴增到x
    x.extend(data)
    
    # data.shape[0]是資料筆數,data.shape[1]是24
    # 第一個檔案資料對應0的索引，n筆x資料就擴增個0
    y.extend([idx] * data.shape[0])

    # 標籤是檔案的主檔名
    label = file.split('.')[0]
    # 將標籤放入labels串列
    labels.append(label)
    

# 將x串列資料轉成numpy格式
x = np.array(x)

# 將y串列資料轉成numpy格式
y = np.array(y)

# 將y做One-Hot Encoder
y=keras.utils.to_categorical(y)

# 將labels串列資料存到labels.txt
with open('labels.txt', 'w') as f:
    for lab in labels:
        f.write(lab + "\n")

# 預設0.75訓練，0.25驗證
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)

# 印出
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# 建立模型
model = keras.Sequential()
# 24個特徵資料
model.add(keras.Input(shape=(24,)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(32, activation='relu'))
# 以len(labels)取得輸出類別數量
model.add(keras.layers.Dense(len(labels),activation='softmax'))

# 顯示模型參數
model.summary()

# 損失函數使用crosse_entropy
model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

# 開始訓練模型
train=model.fit(x_train,y_train,epochs=20,validation_data=(x_test,y_test))

# 儲存模型
model.save('dnn_model.keras')
print('模型儲存完成')
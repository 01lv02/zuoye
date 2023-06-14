import cv2
import dlib
import numpy as np
import streamlit as st
import os
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 添加标题和一些说明文字
st.title("人脸识别加面部修复")
st.write("应用在修复人脸图片")

# 添加一个文本框
text = st.text_input("请在此输入你的文本：")

# 显示文本框的内容
if text:
    st.write("你输入的文本是：", text)
# 定义常量
IMG_SIZE = 64

# 加载人脸检测模型和面部区域标识器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 加载面部修复模型
inpainted_mask_generator = cv2.dnn.readNetFromTensorflow('opencv_face_mask.pb')

# 加载深度学习模型
data_dir = 'lfw/'
X = []
y = []

for person_dir in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_dir)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        # 读取并预处理人脸图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(-1)

        # 将人脸图像和对应的标签添加到 X 和 y 中
        X.append(img)
        y.append(person_dir)

# 将标签编码为数字
labels = list(set(y))
label_dict = {label: i for i, label in enumerate(labels)}
y = [label_dict[label] for label in y]

# 对数据进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 将数据转换为 Numpy 数组
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# 打印数据集的形状和标签
print('X_train shape:', X_train.shape)
print('X_val shape:', X_val.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)
print('y_test shape:', y_test.shape)
print('Number of classes:', len(labels))

# 定义模型
model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dropout(0.5),
    Dense(units=len(labels), activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1), y_train,
                    batch_size=64,
                    epochs=10,
                    validation_data=(X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1), y_val))

# 保存模型
model.save('face_recognition_model.h5')

# 加载模型
loaded_model = tf.keras.models.load_model('face_recognition_model.h5')

# 读取输入图像
img = cv2.imread('input.jpg')

# 将输入图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用人脸检测模型检测人脸并进行面部修复和人脸识别
faces = detector(gray)

for face in faces:
# 使用面部区域标识器获取左右眼和上下嘴唇的坐标
  landmarks = predictor(gray, face)
# 检测黑眼圈并进行修复
  left_eye = landmarks.part(42).x, landmarks.part(43).y
  right_eye = landmarks.part(45).x, landmarks.part(46).y
  eye_width = abs(left_eye[0] - right_eye[0])
  eye_height = eye_width // 2
  left_eye_center = (landmarks.part(45).x, landmarks.part(42).y + eye_height)
  right_eye_center = (landmarks.part(45).x, landmarks.part(45).y + eye_height)

  eye_roi = gray[left_eye_center[1] - eye_height: left_eye_center[1] + eye_height,
          left_eye_center[0] - eye_width // 2: left_eye_center[0] + eye_width // 2]
  eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_GRAY2BGR)
  eye_inpaint_mask = np.zeros_like(eye_roi[:, :, 0])
  cv2.circle(eye_inpaint_mask, (eye_width // 2, eye_height), int(eye_width * 0.4), 1, -1)
  eye_inpaint_mask = cv2.dilate(eye_inpaint_mask, (7, 7), iterations=3)
  eye_inpaint_mask = cv2.GaussianBlur(eye_inpaint_mask, (5, 5), 0)
  eye_inpaint_mask = cv2.normalize(eye_inpaint_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

  eye_inpaint = cv2.inpaint(eye_roi, eye_inpaint_mask, 3, cv2.INPAINT_TELEA)
  img[left_eye_center[1] - eye_height: left_eye_center[1] + eye_height,
  left_eye_center[0] - eye_width // 2: left_eye_center[0] + eye_width // 2] = eye_inpaint

# 检测唇纹并进行修复
mouth = []
for i in range(48, 68):
    mouth.append((landmarks.part(i).x, landmarks.part(i).y))

# 构造唇纹掩模
inpainted_mouth_mask = np.zeros_like(gray, 
                                     dtype=np.float32)
cv2.fillConvexPoly(inpainted_mouth_mask,
                   cv2.convexHull(mouth),
                   1)
inpainted_mouth_mask = inpainted_mouth_mask.astype(np.uint8)
inpainted_mouth_mask = cv2.dnn.blobFromImage(inpainted_mouth_mask)
inpainted_mask_generator.setInput(inpainted_mouth_mask)
inpainted_mask = inpainted_mask_generator.forward()[0][0]
inpainted_mask = cv2.normalize(inpainted_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# 对唇纹进行修复
mouth_inpaint = cv2.inpaint(img, inpainted_mask, 3, cv2.INPAINT_TELEA)

# 进行人脸识别
face_roi = gray[face.top(): face.bottom(), face.left(): face.right()]
face_roi = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
face_roi = face_roi.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
prediction = loaded_model.predict(face_roi)
predicted_label = labels[np.argmax(prediction)]

# 在原始图像中绘制出检测到的人脸、修复后的黑眼圈和唇纹、以及识别出的标签
cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
cv2.putText(img, predicted_label, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Face Detection and Inpainting', img)
cv2.waitKey(0)
#释放所有窗口和资源
cv2.destroyAllWindows()
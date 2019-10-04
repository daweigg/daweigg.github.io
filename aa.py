from datetime import datetime
import cv2
from winsound import Beep
import os

# 人脸识别分类器
faceCascade = cv2.CascadeClassifier(
    r'D:\Programs\PythonVirtualenvs\opencv\Lib\site-packages\cv2\data\haarcascade_fullbody.xml')

# 开启摄像头
cap = cv2.VideoCapture(0)
ok = True

while ok:
    h = datetime.now()
    year = h.year
    month = h.month
    day = h.day
    hour = h.hour
    minute = h.minute
    second = h.second
    name = f'{year}年{month}月{day}日{hour}时{minute}分{second}秒.jpg'

    # 读取摄像头中的图像，ok为是否读取成功的判断参数
    ok, img = cap.read()
    # 转换成灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 人脸检测
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=2,
        minSize=(32, 32)
    )

    # 画矩形
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        Beep(3000, 1000)
        cv2.imwrite('1.jpg', img)
        os.renames('1.jpg', name)

    cv2.imshow('video', img)

    k = cv2.waitKey(1)
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()

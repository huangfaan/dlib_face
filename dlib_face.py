import dlib
import cv2
import numpy as np

class face():
    def __init__(self, img, predictor_path):
        b, g, r = cv2.split(img)
        self.img = cv2.merge([r, b, g])
        self.detector = dlib.get_frontal_face_detector() #获取人脸分类器
        self.predictor = dlib.shape_predictor(predictor_path)  # 获取人脸检测器

    # dlib基于HOG的人脸检测   （还有dlib基于cnn的人脸检测）
    def face_detect(self):
        dets = self.detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))#打印识别到的人脸个数
        for index, face in enumerate(dets):
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index,
                                face.left(), face.top(), face.right(),face.bottom()))

            # 在图片中标注人脸，并显示
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)#线条颜色、宽度

            #标定人脸特征点
            shape = self.predictor(self.img, face)
            # 遍历所有点，打印出其坐标，并用蓝色的圈表示出来
            for index, pt in enumerate(shape.parts()):
                print('Part {}: {}'.format(index, pt))
                pt_pos = (pt.x, pt.y)
                cv2.circle(img, pt_pos, 2, (255, 0, 0), 1)
        return img





img = cv2.imread("2.jpg", cv2.IMREAD_COLOR)
f = face(img, "shape_predictor_68_face_landmarks.dat")
img = f.face_detect()
cv2.imshow("face", img)
cv2.waitKey()
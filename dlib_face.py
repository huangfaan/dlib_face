import dlib
import cv2
import sys
import numpy as np

class face():
    def __init__(self, img, predictor_path, recognition_path):
        b, g, r = cv2.split(img)
        self.img = cv2.merge([r, b, g])
        self.detector = dlib.get_frontal_face_detector() #获取人脸分类器
        #               cnn_face_detector = dlib.cnn_face_detection_model_v1( .dat文件--cnn模型 )
        self.predictor = dlib.shape_predictor(predictor_path)  # 获取人脸特征点检测器
        self.recognition = dlib.face_recognition_model_v1(recognition_path) #可将人脸转为128维向量

    # dlib基于HOG的人脸检测，标定特征点  （还有dlib基于cnn的人脸检测）
    def face_detect(self):
        dets = self.detector(self.img, 1)
        print("Number of faces detected: {}".format(len(dets)))#打印识别到的人脸个数
        if(len(dets) == 0):
            exit()

        for index, face in enumerate(dets):
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index,
                                face.left(), face.top(), face.right(),face.bottom()))

            # 在图片中标注人脸，并显示
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(self.img, (left, top), (right, bottom), (0, 255, 0), 3)#线条颜色、宽度

            #标定人脸特征点
            shape = self.predictor(self.img, face)
            # 遍历所有点，打印出其坐标，并用蓝色的圈表示出来
            for index, pt in enumerate(shape.parts()):
                print('Part {}: {}'.format(index, pt))
                pt_pos = (pt.x, pt.y)
                cv2.circle(self.img, pt_pos, 2, (255, 0, 0), 1)

            face_descriptor = self.recognition.compute_face_descriptor(self.img, shape)  # 计算人脸的128维的向量
            print(face_descriptor)

        return self.img

    #人脸特征点对齐（在这里是将人脸摆正）
    def face_shape_align(self):
        dets = self.detector(self.img, 1)
        print("Number of faces detected: {}".format(len(dets)))#打印识别到的人脸个数
        if(len(dets) == 0):
            exit()

        # 识别人脸特征点，并保存下来
        #//关键点存储方式为full_object_detection，这是一种包含矩形框和关键点位置的数据格式
        shapes = dlib.full_object_detections()
        for det in dets:
            shapes.append(self.predictor(self.img, det))

        #人脸对齐
        #这些面将垂直旋转并缩放到size x size像素或使用可选的指定大小和填充
        images = dlib.get_face_chips(self.img, shapes, size=150)
        # 显示计数，按照这个计数创建窗口
        image_cnt = 0
        # 显示对齐结果
        for image in images:
            image_cnt += 1
            cv_rgb_image = np.array(image).astype(np.uint8)  # 先转换为numpy数组
            cv_bgr_image = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)#opencv下为bgr，故从rgb转换为bgr
            cv2.imshow('%s' % (image_cnt), cv_bgr_image)


#目标跟踪
class myCorrelationTracker():
    def __init__(self, windowName='default window', cameraType=0):
        #自己定义的几个状态
        self.STATUS_RUN_WITHOUT_TRACKER = 0  #不跟踪目标，但是实时显示
        self.STATUS_RUN_WITH_TRACKER = 1  #跟踪目标，实时显示
        self.STATUS_PAUSE = 2  #暂停，卡在当前帧
        self.STATUS_BREAK = 3  #退出
        self.status = self.STATUS_RUN_WITHOUT_TRACKER

        self.track_window = None  # 实时跟踪鼠标的跟踪区域
        self.drag_start = None  # 要检测的物体所在区域
        self.start_flag = True  # 标记，是否开始拖动鼠标

        #创建好显示窗口
        cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(windowName, self.onMouseClicked)
        self.windowName = windowName

        #打开摄像头
        self.cap = cv2.VideoCapture(cameraType)
        #跟踪器
        self.tracker = dlib.correlation_tracker()
        #d当前帧
        self.frame = None

    #按键处理函数
    def keyEventHandler(self):
        keyValue = cv2.waitKey(5) #每隔5秒读取一次按键的健值
        if(keyValue==27): #ESC
            self.status = self.STATUS_BREAK
        if(keyValue==32): #空格
            if(self.status != self.STATUS_PAUSE):
                self.status = self.STATUS_PAUSE
            else:   #再次按下空格时候，重新播发，但是不进行目标识别
                if(self.track_window):
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True
                else:
                    self.status = self.STATUS_RUN_WITHOUT_TRACKER
        if(keyValue==13): #回车
            if(self.status == self.STATUS_PAUSE):  #按下空格之后
                if(self.track_window):   #如果选定了区域，再按回车，表示确定选定区域为跟踪目标
                    self.status = self.STATUS_RUN_WITH_TRACKER
                    self.start_flag = True

    #任务处理函数
    def processHandler(self):
        #不跟踪目标，但是实时显示
        if(self.status == self.STATUS_RUN_WITHOUT_TRACKER):
            ret, self.frame = self.cap.read()
            cv2.imread(self.windowName, self.frame)
        #暂停，暂停时使用鼠标拖动红框，选择目标区域
        elif(self.status == self.STATUS_PAUSE):
            img_first = self.frame.copy() # 不改变原来的帧，拷贝一个新的变量出来
            if(self.track_window):  # 跟踪目标的窗口画出来了，就实时标出来
                cv2.rectangle(img_first, (self.track_window[0], self.track_window[1]),
                              (self.track_window[2], self.track_window[3]), (0,0,255), 1)
            elif(self.selection):  # 跟踪目标的窗口随鼠标拖动实时显示
                cv2.rectangle(img_first, (self.selection[0], self.selection[1]),
                              (self.selection[2], self.selection[3]), (0, 0, 255), 1)
            cv2.imshow(self.windowName, img_first)
        #退出
        elif(self.status == self.STATUS_BREAK):
            self.cap.release() #释放摄像头
            cv2.destroyAllWindows() #释放窗口
            sys.exit()
        elif(self.status == self.STATUS_RUN_WITH_TRACKER):
            ret, self.frame = self.cap.read()  #从摄像头读取一帧
            if(self.start_flag):  #如果是第一帧，需要先初始化
                self.tracker.start_track(self.frame, dlib.rectangle(self.track_window[0],
                            self.track_window[1], self.track_window[2], self.track_window[3]))# 开始跟踪目标
                self.start_flag = False  # 不再是第一帧
            else:
                self.tracker.update(self.frame) #更新
                #得到目标的位置，并显示
                box_predict = self.tracker.get_position()
                cv2.rectangle(self.frame,(int(box_predict.left()),int(box_predict.top())),
                              (int(box_predict.right()),int(box_predict.bottom())),(0,255,255),1)
                cv2.imshow(self.windowName, self.frame)

    # 鼠标点击事件回调函数
    def onMouseClicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
            self.drag_start = (x, y)
            track_window = None
        if self.drag_start:  # 是否开始拖动鼠标，记录鼠标位置
            xMin = min(x, self.drag_start[0])
            yMin = min(y, self.drag_start[1])
            xMax = max(x, self.drag_start[0])
            yMax = max(y, self.drag_start[1])
            self.selection = (xMin, yMin, xMax, yMax)
        if event == cv2.EVENT_LBUTTONUP:  # 鼠标左键松开
            self.drag_start = None
            self.track_window = self.selection
            self.selection = None

    def run(self):
        while(1):
            self.keyEventHandler()
            self.processHandler()



# 换脸 img2的脸替换img1的脸
class transFace():
    def __init__(self, img1, img2, predictor_path):
        self.img1 = img1
        self.img2 = img2
        self.detector = dlib.get_frontal_face_detector() #获取人脸分类器
        #               cnn_face_detector = dlib.cnn_face_detection_model_v1( .dat文件--cnn模型 )
        self.predictor = dlib.shape_predictor(predictor_path)  # 获取人脸特征点检测器

        #颜色校正中所用参数
        self.correct_colors_frac = 0.6
        self.eye_left_point = list(range(42, 48)) #左眼为特征点42-47
        self.eye_right_point = list(range(36, 42)) #右眼为特征点36-41

        #人脸融合中用
        self.brow_left_point = list(range(22, 27))
        self.brow_right_point = list(range(17, 22))
        self.nose_point = list(range(27, 35))
        self.mouth_point = list(range(48, 61))
        self.overlay_points = [self.eye_right_point + self.eye_left_point + self.brow_left_point
                                + self.brow_right_point + self.nose_point + self.mouth_point, ]
        self.feature_amount = 11


    #1、检测人脸并标记特征点
    def get_landmark(self):
        face_rect1 = self.detector(self.img1, 1)
        face_rect2 = self.detector(self.img2, 1)
        if(len(face_rect1) > 1):
            print('Too many faces.We only need no more than one face.')
        elif(len(face_rect1) == 0):
            print('No face.We need at least one face.')
        else:
            print('left {0}; top {1}; right {2}; bottom {3}'.format(face_rect1[0].left(), face_rect1[0].top(),
                                                                    face_rect1[0].right(), face_rect1[0].bottom()))
        if(len(face_rect2) > 1):
            print('Too many faces.We only need no more than one face.')
        elif(len(face_rect2) == 0):
            print('No face.We need at least one face.')
        else:
            print('left {0}; top {1}; right {2}; bottom {3}'.format(face_rect2[0].left(), face_rect2[0].top(),
                                                                    face_rect2[0].right(), face_rect2[0].bottom()))
        landmark1 = np.mat([[p.x, p.y] for p in self.predictor(self.img1, face_rect1[0]).parts()])
        landmark2 = np.mat([[p.x, p.y] for p in self.predictor(self.img2, face_rect2[0]).parts()])
        return landmark1, landmark2


    #2、使用普氏分析法调整脸部 -- 返回仿射变换矩阵
    def trans_from_points(self, points1, points2):
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)
        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T
        #返回仿射变换矩阵。。。np.vstack():在竖直方向上堆叠，np.hstack():在水平方向上平铺
        return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.mat([0., 0., 1.])])

    #2、使用普氏分析法调整脸部 -- 调整脸部
    def warp_image(self, img, M, dshape):
        out_img = np.zeros(dshape)
        out_img = cv2.warpAffine(img, M[:2], (dshape[1], dshape[0]), dst=out_img,
                                 flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT)
        return out_img

    #3、颜色校正,使用img2除以img2的高斯模糊，乘以img1来校正颜色
    def correct_colors(self, img1, img2, landmarks1):
        blur_amount = self.correct_colors_frac * np.linalg.norm(
            np.mean(landmarks1[self.eye_left_point], axis=0) -
            np.mean(landmarks1[self.eye_right_point], axis=0))
        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(img1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(img2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        return (img2.astype(np.float64) * im1_blur.astype(np.float64) /
                im2_blur.astype(np.float64))

    #4、第二张图像特性混合于第一张图像
    # 获取人脸掩模
    def get_face_mask(self, img, landmarks):
        mask = np.zeros(img.shape[:2], dtype=np.float64)
        # 绘制凸包
        points = cv2.convexHull(landmarks[self.overlay_points])  #得到凸包
        cv2.fillConvexPoly(mask, points, color=1) #凸包填充

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
        mask = (cv2.GaussianBlur(mask, (self.feature_amount, self.feature_amount), 0) > 0) * 1.0
        mask = cv2.GaussianBlur(mask, (self.feature_amount, self.feature_amount), 0)
        return mask


    def changed(self):
        landmarks1, landmarks2 = self.get_landmark()
        cv2.imshow("s1", self.img1)
        cv2.imshow("s2", self.img2)
        M = self.trans_from_points(landmarks1, landmarks2)

        img3 = f.warp_image(self.img2, M, self.img1.shape)
        cv2.imshow("3", img3)

        mask = self.get_face_mask(self.img2, landmarks2)
        cv2.imshow("mask", mask)
        warped_mask = self.warp_image(mask, M, self.img1.shape)
        cv2.imshow("warped_mask", warped_mask)
        combined_mask = np.max([self.get_face_mask(self.img1, landmarks1), warped_mask], axis=0)
        cv2.imshow("combined_mask", combined_mask)

        warped_corrected_img = self.correct_colors(self.img1, img3, landmarks1)
        output = self.img1 * (1.0 - combined_mask) + warped_corrected_img * combined_mask
        cv2.normalize(output, output, 0, 1, cv2.NORM_MINMAX)

        cv2.imshow("final", output.astype(output.dtype))
        cv2.waitKey(0)



img1 = cv2.imread("9.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("7.jpg", cv2.IMREAD_COLOR)
f = transFace(img1, img2, predictor_path="shape_predictor_68_face_landmarks.dat")
f.changed()


"""人脸检测和标定
img = cv2.imread("5.jpg", cv2.IMREAD_COLOR)
f = transFace(img, predictor_path="shape_predictor_68_face_landmarks.dat")
mat_landmarks = f.get_landmark()
print(mat_landmarks)
cv2.waitKey()
"""

"""  目标跟踪
if __name__ == '__main__':
    testTracker = myCorrelationTracker(windowName='image', cameraType=1)
    testTracker.run()
"""
import cv2
import cv2.aruco
import numpy as np
import math
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QElapsedTimer,QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot, Qt
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer, Qt
from math import cos, sin
#from test_seg_3 import run
#from TestYolov5_2 import run
from PredictTest_Official_0 import run


class CaptureSignals(QObject):
    newPixmapCaptured = Signal(np.ndarray)
    newParameterCaptured = Signal(np.ndarray)
    newAnimalClassCaptured = Signal(str)

class capture_video(QThread):


    #updateDetail = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.foreGroundIm: cv2.Mat # self.drawFrame = self.foreGroundIm.copy()
        self.frame: cv2.Mat
        self.backGroundIm: cv2.Mat
        self.parameter: cv2.Mat
        self.RealsenseVideo: cv2.VideoCapture
        self.cameraState: bool = False
        self.num = 0
        self.signals = CaptureSignals()
        self.flag = 0
        self.detect_flag = 0
        self.animal_class = str()
        self.timer = QTimer()

    def cvMatToQImage(self,inMat:cv2.Mat) -> QImage:  # update display
        rgb_image = cv2.cvtColor(inMat, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        #print('read frame')
        return q_image
        # else:
        #     print('fail to read frame')


    def connectCamera(self)-> None:
        self.RealsenseVideo = cv2.VideoCapture(0)
        self.RealsenseVideo.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        self.RealsenseVideo.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        # if self.RealsenseVideo.isOpened():
        #     self.cameraState = True
        # print('camera is on')

    def getScreenShot(self) -> None:
        cv2.imwrite('sample/img' + str(self.num) + '.png', self.foreGroundIm)
        print("image saved!")
        self.num += 1

    def disconnectCamera(self) -> None:
        if self.flag == 2:
            self.RealsenseVideo.release()
            self.flag = 0
        # if self.flag == 1:
        #     run(flag=1)
        #     self.flag=0

    def my_output_callback(self, frame):
        if frame is not None:
            self.foreGroundIm = frame
            self.signals.newPixmapCaptured.emit(self.foreGroundIm)
            # print("connect")
            # cv2.imshow("Processed Frame", frame)
            #print('hello')

    def my_parameter_callback(self, parameter):
        if parameter is not None:
            if self.detect_flag == 0:
                self.parameter = np.array(parameter)
                self.signals.newParameterCaptured.emit(self.parameter)

    def my_class_callback(self, animal_class):
        if animal_class is not None:
            if self.detect_flag == 0:
                self.animal_class = animal_class
                self.signals.newAnimalClassCaptured.emit(self.animal_class)

    #@Slot()
    def run(self):
        if self.flag == 1:
            print('thread0')
            while True:
                ret = run(weights='data_segmentv4/best.pt',
                          source=0,
                          max_det=1,
                          image_callback=self.my_output_callback,
                          parameter_callback=self.my_parameter_callback,
                          class_callback=self.my_class_callback,
                          conf_thres=0.70)
                if ret:
                    #print(self.foreGroundIm.dtype)
                    print('emit')
                    self.signals.newPixmapCaptured.emit(self.foreGroundIm)

        elif self.flag == 2:
            print('thread1')
            self.connectCamera()
            parameters = cv2.aruco.DetectorParameters()
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
            # intrinsic_matrix = [[609.3137817382812, 0, 315.90814208984375], [0, 609.192138671875, 240.2545166015625],
            #                     [0, 0, 1]]
            # distortion_coefficients = [0.0, 0.0, 0.0, 0.0, 0.0]
            intrinsic_matrix = [[606.28106689, 0, 327.08996582],
                                [0, 606.4175415, 240.45907593],
                                [0, 0, 1]]
            distortion_coefficients = [0.0, 0.0, 0.0, 0.0, 0.0]
            rvec_samples = []
            tvec_samples = []
            sample_count = 30
            sample_count_1 = 15
            while True:
                ret, self.foreGroundIm = self.RealsenseVideo.read()
                self.frame = self.foreGroundIm.copy()
                corners, ids, rejected_img_points = cv2.aruco.detectMarkers(self.frame, aruco_dict, parameters=parameters)
                if ids is not None:
                    for i in range(len(ids)):
                        #print("detect aruco")
                        cv2.aruco.drawDetectedMarkers(self.frame, corners, ids)
                        # Get the corners of the marker
                        pts = np.int32(corners[i][0])
                        # Draw a border line around the marker
                        cv2.polylines(self.frame, [pts], True, (0, 255, 0), 2)

                        # Estimate the pose of the marker
                        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 65, np.array(intrinsic_matrix, dtype=np.float64),np.array(distortion_coefficients, dtype=np.float64))
                        # Draw the XYZ coordinate system on the marker
                        cv2.drawFrameAxes(self.frame, np.array(intrinsic_matrix, dtype=np.float64),np.array(distortion_coefficients, dtype=np.float64), rvec, tvec, 65,4)
                        #cv2.aruco.drawAxis(self.frame, np.array(intrinsic_matrix, dtype=np.float64),np.array(distortion_coefficients, dtype=np.float64), rvec, tvec, marker_size)

                        # rvec = rvec[i][0]
                        # tvec = tvec[i][0]
                        #
                        # print(f"Marker ID: {ids[i][0]}")
                        # print("Intrinsic Matrix:\n", intrinsic_matrix)
                        # print("Rotation Vector:\n", rvec)
                        # print("Translation Vector:\n", tvec)
                        # print("=" * 50)
                        rvec = np.deg2rad(rvec)
                        rvec, _ = cv2.Rodrigues(rvec) # vector sang matrix
                        tvec = tvec[0][0].reshape(3, 1)
                        #tvec = -np.dot(rvec.T, tvec) # inverse
                        tvec = -tvec

                        if cv2.waitKey(1) & 0xFF == ord('c'):
                            rvec_samples.append(rvec)
                            tvec_samples.append(tvec)
                            print(f"Sample {len(rvec_samples)} saved.")

                            # Kiểm tra nếu đủ 35 mẫu
                            if len(rvec_samples) >= sample_count:
                                np.savez("camera2aruco_samples", rvec_samples=rvec_samples, tvec_samples=tvec_samples)
                                print("30 samples collected. saved")

                                # Khởi tạo ma trận kết quả ban đầu cho rvec và tvec
                                # rvec_product = np.eye(3)  # Ma trận đơn vị 3x3
                                # tvec_product = np.zeros((1, 3))  # Bắt đầu với ma trận 1x3 giá trị 0
                                #
                                # # Nhân từng ma trận rvec và tvec trong danh sách với ma trận kết quả
                                # for r, t in zip(rvec_samples, tvec_samples):
                                #     #r_mat, _ = cv2.Rodrigues(r)  # Chuyển đổi rvec sang ma trận xoay 3x3
                                #     rvec_product = np.matmul(rvec_product, r)  # Nhân liên tiếp các ma trận xoay
                                #     tvec_product = np.matmul(tvec_product, r) + t.reshape(1, 3)  # Cập nhật tvec
                                #
                                # print("Product of Rotation Vectors (rvec):\n", rvec_product)
                                # print("Product of Translation Vectors (tvec):\n", tvec_product)

                                #np.savez('calib_aruco_data', rvec_product=rvec_product, tvec_product=tvec_product)


                                # Xóa các mẫu sau khi tính toán
                                rvec_samples.clear()
                                tvec_samples.clear()
                        self.signals.newPixmapCaptured.emit(self.frame)
                else:
                    self.signals.newPixmapCaptured.emit(self.foreGroundIm)
                cv2.imshow('aruco_frame', self.frame)

###############################################################
#self.backGroundIm = cv2.imread("src\\../Image/Background.jpg")
#self.newPixmapCaptured.emit(self.foreGroundIm)


# class VideoCaptureThread(QThread):
#     frameCaptured = Signal(QImage)
#
#     def __init__(self, camera_index=0):
#         super().__init__()
#         self.camera_index = camera_index
#         self.capture = None
#         self.running = False
#         self.timer = QTimer()
#         self.timer.timeout.connect(self.update_frame)
#
#     def start_capture(self):
#         self.capture = cv2.VideoCapture(self.camera_index)
#         if self.capture.isOpened():
#             self.running = True
#             self.timer.start(30)
#         #     print("Camera started.")
#         # else:
#         #     print("Failed to start camera.")
#
#     def update_frame(self):
#         if not self.running:
#             return
#         ret, frame = self.capture.read()
#         if ret:
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h, w, ch = rgb_frame.shape
#             bytes_per_line = ch * w
#             q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
#             #print("Emitting frameCaptured signal")
#             self.frameCaptured.emit(q_image)
#         # else:
#         #     print("Failed to read frame.")
#
#     def run(self):
#         self.start_capture()
#         while self.running:
#             self.update_frame()  # Directly call update_frame in the loop
#             self.msleep(30)  # Delay for frame rate control (30 ms for ~30 FPS)
#             #self.exec_()  # Keep the thread running and listening for events (such as timer events)
#
#     def stop_capture(self):
#         self.running = False
#         self.timer.stop()
#         if self.capture and self.capture.isOpened():
#             self.capture.release()
#         print("Camera stopped.")
#
#     def __del__(self):
#         self.stop_capture()

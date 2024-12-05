import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QObject, QThread, pyqtSignal as Signal, pyqtSlot as Slot
from PyQt5.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QHeaderView,
    QMessageBox,
    QRadioButton,
    QSlider,
    QFrame,
    QTableWidgetItem,
    QAbstractSlider,
    QGroupBox
)
import numpy as np
from typing import *
#from testrun_gripper import *
from testrun import *
from structure import *
from camera_QT import *
from moto import *
from Arduino import *
import time
import sys,os
#from arduino_thesis import *
import asyncio
#import convert_deg_rad as tool
#import transforms3d as tfs
from scipy.interpolate import interp1d


xyz_world_pos = np.array([[0],[0],[0]])
class GUI(Ui_MainWindow):
    def __init__(self,MainWindow) -> None:
        super(GUI,self).__init__()
        self.setupUi(MainWindow)
        self.initVariable()      
        self.initCallback()

    def initVariable(self) -> None:
        self.connection = Motomini() # after this when we call self.connection.(any_deft) from motonmini() class
        self.arduino = ArduinoController()
        self.data = cordinateVariable()
        self.flagBuffer: list[any]
        self.main_array = np.empty([])
        self.describe = txHeader()
        self.camera = capture_video()
        #self.camera_thread = VideoCaptureThread() #Initialize camera and camera thread
        self.counter = 0
        self.counter_frame = 0
        #self.Arduino_state: bool = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_State)
        self.timer1 = QTimer()
        self.timer1.timeout.connect(self.GetPosition)
        self.timer2 = QTimer()
        #self.timer2.timeout.connect(self.pick_and_place)
        self.timer3 = QTimer()
        # self.timer3.timeout.connect(self.Pick)
        # self.timer3.start(100)
        # self.extrinsic_Rmatrix = np.array([[ 0.02916868, -0.99876418,  0.04024043],
        #                                     [-0.99914049, -0.02794617,  0.03061531],
        #                                     [-0.02945291, -0.04109885, -0.99872089]])
        # self.extrinsic_Tmatrix = np.array([[ 208.100702],#215
        #                                     [-205.49859565],#-205
        #                                     [ 244.16532057]])
        self.extrinsic_Rmatrix = np.array([[-0.3223689996337005, 0.9194664687996084, 0.2250769664544691],
                                            [0.691285793679262, 0.3911001196502168, -0.6075892098011773],
                                            [-0.6466855337274835, -0.04027541635743781, -0.7616926619738664]])
        self.extrinsic_Tmatrix = np.array([[0.2674104793664804],
                                            [-0.08410275183689284],
                                            [0.1615711395358033]])
        self.extrinsic_Tmatrix = self.extrinsic_Tmatrix * 1000
        # [-0.3223689996337005, 0.9194664687996084, 0.2250769664544691;
        # 0.691285793679262, 0.3911001196502168, -0.6075892098011773;
        # -0.6466855337274835, -0.04027541635743781, -0.7616926619738664]
        # [0.2674104793664804;
        # -0.08410275183689284;
        # 0.1615711395358033]
        # self.extrinsic = np.array([[ 0.02916868, -0.99876418,  0.04024043, 215.100702],
        #                            [-0.99914049, -0.02794617,  0.03061531, -205.49859565],
        #                            [-0.02945291, -0.04109885, -0.99872089, 244.16532057],
        #                            [0          ,  0         ,  0         , 1            ]])
        self.sample = 30
        self.sample_1 = 15
        self.rt_combined = []
        self.t_sample = []
        self.Matrix_pos = []
        self.Matrix_angle = []
        self.stage = 0
        self.xyz_world = []
        self.homo_rot = []
        self.animal_class = str()
        #Init for pick and place object
        self.object_flag = 0
        self.length = 0
        self.current_class = None
        self.xpos = 0
        self.state = 0
        self.count_1 = 0
        self.count_2 = 0
        #Set up position for arrangement
        self.z_h = -15
        self.z_l = -25
        ########################RED##############################
        #Small
        self.xrs = 14.477
        self.yrs = 152.570
        #Big
        self.xrb = 14.477
        self.yrb = 210.172
        ########################GREEN##############################
        #Small
        self.xgs = 143.642
        self.ygs = 130.570
        #Big
        self.xgb = 143.642
        self.ygb = 200.172
        ########################BLUE##############################
        #Small
        self.xbls = 235.509
        self.ybls = 130.570
        #Big
        self.xblb = 235.500
        self.yblb = 180.172
    def initCallback(self) -> None:
        self.connect_Button.clicked.connect(self.checkConnect)
        self.servo_on.clicked.connect(self.ctrlServoCallback)
        self.Go_Position.clicked.connect(self.CartasianMove)
        self.Go_Home.clicked.connect(self.GoHome)
        self.Get_Position.clicked.connect(self.GetPosition)
        self.Get_Pulse.clicked.connect(self.GetPulse)
        self.Distance_mm.valueChanged.connect(self.ShowDistancemm)
        self.Distance_deg.valueChanged.connect(self.ShowDistancedeg)
        self.Speed_mm.valueChanged.connect(self.ShowSpeedmm)
        self.Speed_deg.valueChanged.connect(self.ShowSpeeddeg)
        self.X_Incre.clicked.connect(self.XINCRE)
        self.Y_Incre.clicked.connect(self.YINCRE)
        self.Z_Incre.clicked.connect(self.ZINCRE)
        self.Roll_Incre.clicked.connect(self.ROLLINCRE)
        self.Pitch_Incre.clicked.connect(self.PITCHINCRE)
        self.Yall_Incre.clicked.connect(self.YALLINCRE)
        self.X_Decre.clicked.connect(self.XDECRE)
        self.Y_Decre.clicked.connect(self.YDECRE)
        self.Z_Decre.clicked.connect(self.ZDECRE)
        self.Roll_Decre.clicked.connect(self.ROLLDECRE)
        self.Pitch_Decre.clicked.connect(self.PITCHDECRE)
        self.Yall_Decre.clicked.connect(self.YALLDECRE)
        self.radioButton.clicked.connect(self.CartasianStraightState)
        self.radioButton_2.clicked.connect(self.CartasianLinkState)
        self.radioButton_3.clicked.connect(self.PulseState)
        self.Camera_On.clicked.connect(self.start_capture_video)
        # self.camera_thread.frameCaptured.connect(self.display_camera_frame)
        self.camera.signals.newPixmapCaptured.connect(self.update_image) # Kết nối tín hiệu
        self.camera.signals.newParameterCaptured.connect(self.update_parameter)
        self.camera.signals.newAnimalClassCaptured.connect(self.update_class)
        self.Take_Calib.clicked.connect(self.take_sample_calib)
        self.Stop_Calib.clicked.connect(self.export_set_of_point)
        self.YOLO_On.clicked.connect(self.start_detection)
        self.Connect_Arduino.clicked.connect(self.connect_arduino)
        self.Grab.clicked.connect(self.send_angle)
        self.Pick_Place.clicked.connect(self.start_pick_and_place)
        # self.Pick_Place.clicked.connect(self.Pick)
        #self.Camera_Off.clicked.connect(self.stop_capture_video)
        #self.CartasianIncre.clicked.connect(self.CartasianMoveIncre)
        #self.CartasianDecre.clicked.connect(self.CartasianMoveDecre)
        #self.slider()
        self.thread = {}

    def hom_2_cart(self, homogenoeus_matrix):
        HT = np.array(homogenoeus_matrix[0:3, 3])
        HR = np.rad2deg(tfs.euler.mat2euler(homogenoeus_matrix[0:3, 0:3]))
        cartesian_matrix = np.array([*HT, *HR], np.float64)
        return cartesian_matrix

    ###########################################################################################################################################
    def cart_2_hom(self, cartesian_matrix):
        CT = np.array(cartesian_matrix[0:3]).reshape(3, 1)
        CR = tfs.euler.euler2mat(math.radians(cartesian_matrix[3]),
                                 math.radians(cartesian_matrix[4]),
                                 math.radians(cartesian_matrix[5]))
        homogenoeus_matrix = np.row_stack((np.column_stack((CR, CT)), np.array([0, 0, 0, 1])))
        return homogenoeus_matrix

    #########################################UDP CONNECTION################################################
    def checkConnect(self):
        try:
            if (self.connection.connectState == False):
                self.connection.connectMotomini(self.edit_ip.toPlainText(), int(self.edit_port.toPlainText()))
                if self.connection.checkConnectState() == False:
                    self.messBox("Warning!!!", "NO connection with robot")
                if self.connection.checkConnectState() == True:
                    self.messBox("Sucessfully!!!", "YES connection with robot")
                    self.connect_button.setText("Disconnection")
                    self.connect_button.setStyleSheet(
                        "background-color: red; color: #F9F9F9;text-align: center; font-size: 16px; font-weight: bold; border-radius: 6px")
                    self.timer.start(100)

            elif self.connect_button.text() == "Disconnection":
                self.connection.disconnectMotomini()
                self.connect_button.setText("Connection")
                self.connect_button.setStyleSheet(
                    "background-color: green; color: #F9F9F9;text-align: center; font-size: 16px; font-weight: bold; border-radius: 6px")
                self.timer.stop()
        except Exception as e:
            print(f"An error occurred: {e}")

    def checkServo(func):
        def check(self):
            if self.connection.checkServoState() == False:
                self.messBox("Warning!!!", "Servo is off")
                return
            return func(self)
        return check

    #@checkConnect
    def ctrlServoCallback(self) -> None:
        if(self.connection.checkServoState() == False):
            self.connection.onServo()
            self.servo_on.setText("ON SERVO")
            self.servo_on.setStyleSheet("QPushButton {color: red;font-size: 14px; font-weight: bold}}")
            self.Status.setText("STATE: ON")
            self.Status.setStyleSheet("QPushButton {color: green;font-size: 14px; font-weight: bold}")
            # self.Status.setStyleSheet(
            #     "QPushButton {color: green; font-size: 16px; font-weight: bold; background-color: lightgreen;}")
            #This will be the code to active the button,slider,textbox,...
            self.groupBox_2.setEnabled(True)
            self.groupBox_3.setEnabled(True)
            self.groupBox_4.setEnabled(True)
            self.groupBox_5.setEnabled(True)
            #self.groupBox_6.setEnabled(True)
            self.groupBox_7.setEnabled(True)
            self.Pick_Place.setEnabled(True)
            #
            self.showBytes()
        else:
            self.connection.offServo()
            self.groupBox_2.setEnabled(False)
            self.groupBox_3.setEnabled(False)
            self.groupBox_4.setEnabled(False)
            self.groupBox_5.setEnabled(False)
            #self.groupBox_6.setEnabled(False)
            self.groupBox_7.setEnabled(False)
            self.Pick_Place.setEnabled(False)
            self.showBytes()
            self.servo_on.setText("OFF SERVO")
            self.servo_on.setStyleSheet("QPushButton {color: green;}")
            self.Status.setText("STATE: OFF")
            self.Status.setStyleSheet("QPushButton {color: black;}")
            #print(self.counter)
        return

    #############################################SERIAL PORT############################################
    def connect_arduino(self)->None:
        if self.arduino.arduinoState == False and self.Connect_Arduino.text() == "Connect":
            serial_port = self.edit_port_com.toPlainText()
            baud_rate = int(self.edit_baud_rate.toPlainText())
            self.arduino.connect_to_arduino(serial_port, baud_rate)
            self.arduino.arduinoState = True
            #self.Connect_Arduino.setText("Disconnect")
            self.Connect_Arduino.setText("Disconnect")
            self.Connect_Arduino.setStyleSheet(
            "background-color: red; color: #F9F9F9;text-align: center; font-size: 11px; font-weight: bold; border-radius: 6px")
            self.Arduino_state.setText("STATE: ON")
            self.Status.setStyleSheet("QPushButton {color: green;font-size: 14px; font-weight: bold}")
        elif self.arduino.arduinoState == True and self.Connect_Arduino.text() == "Disconnect":
            self.arduino.disconnect_to_arduino()
            self.Connect_Arduino.setText("Connect")
            self.Connect_Arduino.setStyleSheet(
                "background-color: Green; color: #F9F9F9;text-align: center; font-size: 13px; font-weight: bold; border-radius: 6px")
            self.Arduino_state.setText("STATE: OFF")
            self.Status.setStyleSheet("QPushButton {color: black;}")
            self.arduino.arduinoState = False

    def send_angle(self):
        self.arduino.command = self.edit_angle.toPlainText()
        self.arduino.send_command_seal(self.arduino.command)

    def send_grasp_angle_seal(self, value: str): # auto
        # print("value",value)
        self.arduino.command = value
        self.arduino.send_command_seal(self.arduino.command)

    def send_grasp_angle_duck(self, value: str): # auto
        # print("value",value)
        self.arduino.command = value
        self.arduino.send_command_duck(self.arduino.command)

    ###########################################READ RUNNING AND STANDING################################
    def read_State(self):
        self.connection.state_Robot()
        self.showBytes()
        if (self.connection.rx_buffer[32] == 194):
            self.Receive_Data.setText("Standing")
            self.Receive_Data.setStyleSheet(
                "background-color: red; color: #F9F9F9; text-align: center; font-size: 16px; font-weight: bold; border-radius: 6px")
            self.timer1.stop()
        elif (self.connection.rx_buffer[32] == 202):
            self.Receive_Data.setText("Running")
            self.Receive_Data.setStyleSheet(
                "background-color: green; color: #F9F9F9;text-align: center; font-size: 16px; font-weight: bold; border-radius: 6px")
            self.timer1.start(100)
        # self.Receive_data.setText(str(self.connection.rx_buffer[32]))
        # if(self.Receive_data.text()=="Standing"):
        #         self.send_auto("90")
        #     if(self.Receive_data.text()=="Running"):
        #         self.send_auto("0")

    def showBytes(self) -> None:
        #self.recieve_buffer = self.connection.rx_buffer[0:32]

        #self.a = QByteArray("0x59\0x45\0x52\0x43\0x20\0x00")
        #self.connection.rx_buffer.replace[0:5] =  ["0x59\0x4\0x52\0x43\0x20\0x00"]
        #self.connection.rx_buffer[12:24] = [ 0x80, 0x00, 0x00, 0x00, 0x39, 0x39 ,0x39 ,0x39 ,0x39 ,0x39 ,0x39 ,0x39]
        self.Recieving_Byte.setText(str(self.connection.rx_buffer[0:520]))
        #print(str(self.connection.rx_buffer[0:520]))
        return

    def CartasianStraightState(self) -> None:
        self.label_4.setText("X (mm)")
        self.label_5.setText("Y (mm)")
        self.label_6.setText("Z (mm)")
        self.label_7.setText("Pitch (deg)")
        self.label_8.setText("Roll (deg)")
        self.label_9.setText("Speed(mm/s)")
        self.label_10.setText("Yaw (deg)")
        self.Go_Position.setText("Go Cartesian")
        return

    def CartasianLinkState(self) -> None:
        self.label_4.setText("X (mm)")
        self.label_5.setText("Y (mm)")
        self.label_6.setText("Z (mm)")
        self.label_7.setText("Pitch (deg)")
        self.label_8.setText("Roll (deg)")
        self.label_9.setText("Speed(%)")
        self.label_10.setText("Yaw (deg)")
        self.Go_Position.setText("Go Cartesian")
        return

    def PulseState(self) -> None:
        self.label_4.setText("S (deg)")
        self.label_5.setText("L (deg)")
        self.label_6.setText("U (deg)")
        self.label_7.setText("B (deg)")
        self.label_8.setText("R (deg)")
        self.label_9.setText("Speed")
        self.label_10.setText("T (deg)")
        self.Go_Position.setText("Go Pulse")
        return

    #This is RadioButtonPart
    def CartasianMove(self) -> None:
        a = int(float(self.X.toPlainText())*1000)
        b = int(float(self.Y.toPlainText())*1000)
        c = int(float(self.Z.toPlainText())*1000)
        d = int(float(self.Roll.toPlainText())*10000)
        e = int(float(self.Pitch.toPlainText())*10000)
        f = int(float(self.Yaw.toPlainText())*10000)
        #check if the data is taking correctly
        print(a,b,c,d,e,f)

        if (self.radioButton_3.isChecked()==True):
            self.connection.movePulsePos(int(self.Speed.toPlainText()),int(self.X.toPlainText()),int(self.Y.toPlainText()),int(self.Z.toPlainText()),int(self.Roll.toPlainText()),int(self.Pitch.toPlainText()),int(self.Yaw.toPlainText()))
                #self.connection.movePulsePos(int(self.Speed.toPlainText()),int(self.X.toPlainText()),int(self.Y.toPlainText()),int(self.Z.toPlainText()),int(self.Roll.toPlainText()),int(self.Pitch.toPlainText()),int(self.Yaw.toPlainText()))
                #int.pos[0](self.X.toPlainText()),int.pos[1](self.Y.toPlainText()),int.pos[2](self.Z.toPlainText()),int.pos[3](self.Roll.toPlainText()),int.pos[4](self.Pitch.toPlai)
                #int(self.X.toPlainText()),int(self.Y.toPlainText()),int(self.Z.toPlainText()),int(self.Roll.toPlainText()),int(self.Pitch.toPlainText()),int(self.Yaw.toPlainText())
                #int.pos[0](self.X.toPlainText()),int.pos[1](self.Y.toPlainText()),int.pos[2](self.Z.toPlainText()),int.pos[3](self.Roll.toPlainText()),int.pos[4](self.Pitch.toPlainText()),int.pos[5](self.Yaw.toPlainText())
            self.Go_Position.setText("ON Action")
            self.Go_Position.setStyleSheet("QPushButton {color: green;}")
            self.Go_Home.setStyleSheet("QPushButton {color: black;}")
            self.Get_Position.setText("OFF Active")
            self.Get_Position.setStyleSheet("QPushButton {color: gray;}")
            self.Get_Pulse.setText("OFF Active")
            self.Get_Pulse.setStyleSheet("QPushButton {color: gray;}")
        elif (self.radioButton_2.isChecked()==True):
            self.connection.moveCartasianPos(int(self.Speed.toPlainText()),a,b,c,d,e,f)
            self.showBytes()
            self.Go_Position.setText("ON Action")
            self.Go_Position.setStyleSheet("QPushButton {color: green;}")
            self.Go_Home.setStyleSheet("QPushButton {color: black;}")
            self.Get_Position.setText("OFF Active")
            self.Get_Position.setStyleSheet("QPushButton {color: gray;}")
            self.Get_Pulse.setText("OFF Active")
            self.Get_Pulse.setStyleSheet("QPushButton {color: gray;}")
        elif (self.radioButton.isChecked()==True):
            self.connection.moveCartasianStraight(int(self.Speed.toPlainText()),a,b,c,d,e,f)
            self.showBytes()
            self.Go_Position.setText("ON Action")
            self.Go_Position.setStyleSheet("QPushButton {color: green;}")
            self.Go_Home.setStyleSheet("QPushButton {color: black;}")
            self.Get_Position.setText("OFF Active")
            self.Get_Position.setStyleSheet("QPushButton {color: gray;}")
            self.Get_Pulse.setText("OFF Active")
            self.Get_Pulse.setStyleSheet("QPushButton {color: gray;}")
        else:
            self.Go_Position.setText("OFF Action")
            self.Go_Position.setStyleSheet("QPushButton {color: black;}")
        return
  

    def ShowDistancemm(self):
        self.Show_distance.setText(str(self.Distance_mm.value()))
        return
    def ShowDistancedeg(self):
        self.Show_deg.setText(str(self.Distance_deg.value()))
        return
    def ShowSpeedmm(self):
        self.Show_speed_mm.setText(str(self.Speed_mm.value()))
        return
    def ShowSpeeddeg(self):
        self.Show_speed_deg.setText(str(self.Speed_deg.value()))
        return
     
    def XINCRE(self):
        self.speed = int(self.Show_speed_mm.toPlainText())
        self.showBytes()
        self.x  = int(self.Show_distance.toPlainText()) #Incre X amount
        #print(self.x)
        self.connection.moveCartasianPosIncre(self.speed, self.x, 0, 0, 0, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def YINCRE(self):
        self.speed = int(self.Show_speed_mm.toPlainText())
        self.showBytes()
        self.y  = int(self.Show_distance.toPlainText()) #Incre Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, self.y, 0, 0, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def ZINCRE(self):
        self.speed = int(self.Show_speed_mm.toPlainText())
        self.showBytes()
        self.z  = int(self.Show_distance.toPlainText()) #Incre Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, self.z, 0, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def ROLLINCRE(self):
        self.speed = int(self.Show_speed_deg.toPlainText())
        self.showBytes()
        self.m  = int(self.Show_deg.toPlainText()) #Incre Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, 0, self.m, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def PITCHINCRE(self):
        self.speed = int(self.Show_speed_deg.toPlainText())
        self.showBytes()
        self.n  = int(self.Show_deg.toPlainText()) #Incre Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, 0, 0, self.n, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def YALLINCRE(self):
        self.speed = int(self.Show_speed_deg.toPlainText())
        self.showBytes()
        self.o  = int(self.Show_deg.toPlainText()) #Incre Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, 0, 0, 0, self.o)
        self.GetPosition()
        self.GetPulse()
        return

    def XDECRE(self):
        self.speed = int(self.Show_speed_mm.toPlainText())
        self.showBytes()
        self.x  = -int(self.Show_distance.toPlainText()) #DECRE X amount

        self.connection.moveCartasianPosIncre(self.speed, self.x, 0, 0, 0, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def YDECRE(self):
        self.speed = int(self.Show_speed_mm.toPlainText())
        self.showBytes()
        self.y  = -int(self.Show_distance.toPlainText()) #DECRE Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, self.y, 0, 0, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def ZDECRE(self):
        self.speed = int(self.Show_speed_mm.toPlainText())
        self.showBytes()
        self.z  = -int(self.Show_distance.toPlainText()) #DECRE Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, self.z, 0, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def ROLLDECRE(self):
        self.speed = int(self.Show_speed_deg.toPlainText())
        self.showBytes()
        self.m  = -int(self.Show_deg.toPlainText()) #Incre Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, 0, self.m, 0, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def PITCHDECRE(self):
        self.speed = int(self.Show_speed_deg.toPlainText())
        self.showBytes()
        self.n  = -int(self.Show_deg.toPlainText()) #Incre Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, 0, 0, self.n, 0)
        self.GetPosition()
        self.GetPulse()
        return
    def YALLDECRE(self):
        self.speed = int(self.Show_speed_deg.toPlainText())
        self.showBytes()
        self.o  = -int(self.Show_deg.toPlainText()) #DECRE Y amount

        self.connection.moveCartasianPosIncre(self.speed, 0, 0, 0, 0, 0, self.o)
        self.GetPosition()
        self.GetPulse()
        return


    def GetPosition(self) -> None:
        if(self.connection.checkServoState() == True):
            self.connection.getVariablePos()
            self.showBytes()
            self.X_POS.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[52:55],"little", signed=True)))
            self.Y_POS.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[56:59],"little", signed=True)))
            self.Z_POS.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[60:63],"little", signed=True)))
            self.Roll_POS.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[64:67],"little", signed=True)))
            self.Pitch_POS.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[68:71],"little", signed=True)))
            self.Yaw_POS.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[72:75],"little", signed=True)))
            
            #self.X_Incre.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[52:55],"little", signed=True)))
            #self.Y_Incre.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[56:59],"little", signed=True)))
            #self.Z_Incre.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[60:63],"little", signed=True)))
            #self.Roll_Incre.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[64:67],"little", signed=True)))
            #self.Pitch_Incre.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[68:71],"little", signed=True)))
            #self.Yall_Incre.setText(str(int.from_bytes(self.connection.rx_buffer_cartesian[72:75],"little", signed=True)))
            #set(int.from_bytes(self.rx_buffer_cartesian[52:55])), int.from_bytes(self.rx_buffer_cartesian[56:59]), int.from_bytes(self.rx_buffer_cartesian[60:63]), int.from_bytes(self.rx_buffer_cartesian[64:67]), int.from_bytes(self.rx_buffer_cartesian[68:71]), int.from_bytes(self.rx_buffer_cartesian[72:75]))
            self.Get_Position.setText("ON Active")
            self.Get_Position.setStyleSheet("QPushButton {color: cyan;}")
            self.Go_Position.setText("OFF Action")
            self.Go_Position.setStyleSheet("QPushButton {color: black;}")
            self.Get_Pulse.setText("OFF Active")
            self.Get_Pulse.setStyleSheet("QPushButton {color: gray;}")
            self.showBytes()
        else:
            self.Get_Position.setText("OFF Active")
            self.Get_Position.setStyleSheet("QPushButton {color: gray;}")
        return

    def GetPulse(self) -> None:
        if(self.connection.checkServoState() == True):
            self.connection.getVariablePulse()
            self.S_POS.setText(str(round(int.from_bytes(self.connection.rx_buffer_pulse[52:55],"little", signed=True)/self.data.S_pulse_degree)))
            self.L_POS.setText(str(round(int.from_bytes(self.connection.rx_buffer_pulse[56:59],"little", signed=True)/self.data.L_pulse_degree)))
            self.U_POS.setText(str(round(int.from_bytes(self.connection.rx_buffer_pulse[60:63],"little", signed=True)/self.data.U_pulse_degree)))
            self.R_POS.setText(str(round(int.from_bytes(self.connection.rx_buffer_pulse[64:67],"little", signed=True)/self.data.RBT_pulse_degree)))
            self.B_POS.setText(str(round(int.from_bytes(self.connection.rx_buffer_pulse[68:71],"little", signed=True)/self.data.RBT_pulse_degree)))
            self.T_POS.setText(str(round(int.from_bytes(self.connection.rx_buffer_pulse[72:75],"little", signed=True)/self.data.RBT_pulse_degree)))
            self.showBytes()
            #set(int.from_bytes(self.rx_buffer_cartesian[52:55])), int.from_bytes(self.rx_buffer_cartesian[56:59]), int.from_bytes(self.rx_buffer_cartesian[60:63]), int.from_bytes(self.rx_buffer_cartesian[64:67]), int.from_bytes(self.rx_buffer_cartesian[68:71]), int.from_bytes(self.rx_buffer_cartesian[72:75]))
            self.Get_Pulse.setText("ON Active")
            self.Get_Pulse.setStyleSheet("QPushButton {color: orange;}")
            self.Get_Position.setText("OFF Active")
            self.Get_Position.setStyleSheet("QPushButton {color: gray;}")
            self.Go_Position.setText("OFF Action")
            self.Go_Position.setStyleSheet("QPushButton {color: black;}")
        else:
            self.Get_Pulse.setText("OFF Active")
            self.Get_Pulse.setStyleSheet("QPushButton {color: gray;}")
        return

    def GoHome(self) -> None:
        if(self.connection.checkServoState() == True):
            self.connection.movePulsePos(2500,0,0,0,0,0,0)
            self.Go_Home.setStyleSheet("QPushButton {color: green;}")
            self.Go_Position.setText("OFF ACtion")
            self.Go_Position.setStyleSheet("QPushButton {color: black;}")
        else:
            self.Go_Home.setStyleSheet("QPushButton {color: black;}")
        return

    def messBox(self,tilte: str, text: str):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(text)
        msgBox.setWindowTitle(tilte)
        msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msgBox.exec()

    def take_sample_calib(self): # Take_Calib
        if not self.camera.isRunning():
            self.camera.flag = 2
            self.camera.start()

    def export_set_of_point(self): # Stop_Calib
        #self.camera.disconnectCamera()
        self.calculate_calib()

    def robot_matrices(self):
        roll = int.from_bytes(self.connection.rx_buffer_cartesian[64:67],"little", signed=True)
        pitch = int.from_bytes(self.connection.rx_buffer_cartesian[68:71],"little", signed=True)
        yaw = int.from_bytes(self.connection.rx_buffer_cartesian[72:75],"little", signed=True)
        Rx = [[1, 0, 0],
              [0, cos(roll), -sin(roll)],
              [0, sin(roll), cos(roll)]]

        Ry = [[cos(pitch), 0, sin(pitch)],
              [0, 1, 0],
              [-sin(pitch), 0, cos(pitch)]]

        Rz = [[cos(yaw), -sin(yaw), 0],
              [sin(yaw), cos(yaw), 0],
              [0, 0, 1]]
        rotation_combined = np.matmul(np.matmul(Rx, Ry), Rz)

        T = [[int.from_bytes(self.connection.rx_buffer_cartesian[52:55], "little", signed=True)],
             [int.from_bytes(self.connection.rx_buffer_cartesian[56:59], "little", signed=True)],
             [int.from_bytes(self.connection.rx_buffer_cartesian[60:63], "little", signed=True)]]
        T = np.array(T).reshape(3, 1)

        self.rt_combined.append(rotation_combined)
        #print(self.rt_combined)
        self.t_sample.append(T)
        #print('t_sample', self.t_sample)
        print(len(self.rt_combined))

        if len(self.rt_combined) >= self.sample:
            np.savez("camera2robot_samples", rt_combined=self.rt_combined, t_sample=self.t_sample)
            print("saved")

            self.rt_combined.clear()
            self.t_sample.clear()
        # print('Rx', Rx)
        # print('Ry', Ry)
        # print('Rz', Rz)
        # print('rotation_combined', rotation_combined)
        # print('T', T)
        # return Rx, Ry, Rz, T

    def calculate_calib(self):
        # load data robot
        data = np.load("camera2robot_samples.npz")
        rt_combined_load = data["rt_combined"]
        rt_combined_load = np.deg2rad(rt_combined_load)
        print(rt_combined_load)
        t_sample_load = data["t_sample"]
        #t_sample_load = np.array([[value / 1000 for value in row] for row in t_sample_load])  # mm
        print(t_sample_load)

        # load data aruco
        data1 = np.load("camera2aruco_samples.npz")
        rvec_samples_loaded = data1["rvec_samples"]
        # Convert rvec_samples_loaded to rotation matrices if necessary
        R_cam_samples = np.array([cv2.Rodrigues(rvec)[0] for rvec in rvec_samples_loaded])
        print(rvec_samples_loaded)
        tvec_samples_loaded = data1["tvec_samples"]
        tvec_samples_loaded = np.array([np.linalg.pinv(tvec) for tvec in tvec_samples_loaded])
        print(tvec_samples_loaded)

        # Tính toán ma trận chuyển đổi camera sang gripper
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            rt_combined_load, t_sample_load,
            rvec_samples_loaded, tvec_samples_loaded, method=cv2.CALIB_HAND_EYE_TSAI)

        print("Ma trận xoay từ camera sang gripper (R_cam2gripper):")
        print(R_cam2gripper)
        print("Vector dịch chuyển từ camera sang gripper (t_cam2gripper):")
        print(t_cam2gripper)

    def start_capture_video(self):
        if not self.camera.isRunning():
            #self.camera.start()
            self.camera.connectCamera()
        # self.robot_matrices()

    def stop_capture_video(self):
        if self.camera.isRunning():
            self.camera.disconnectCamera()
            self.camera.quit()
            self.label_28.clear()

    def start_detection(self): # Yolo_On
        if not self.camera.isRunning():
            self.camera.flag = 1
            self.camera.start()

    def update_image(self, image: np.ndarray):
        qt_image = self.camera.cvMatToQImage(image)
        self.label_28.setPixmap(QPixmap.fromImage(qt_image))

    def update_class(self, animal_class):
        self.animal_class = animal_class
        # print("animal", self.animal_class)

    def update_parameter(self, parameter):
        print("Received parameter data:", parameter)
        # nhan ma tran calib (4x4) voi ma tran vi tri vat(4x1)
        X = Y = Z = 0

        if float(parameter[0]) != 0 and float(parameter[1]) != 0 and float(parameter[2]) != 0:
            if float(parameter[2]) < 400:
                X = parameter[0]
                Y = parameter[1]
                Z = parameter[2]
        Roll = parameter[3]
        Pitch = parameter[4]
        Yaw = parameter[5]
        self.extrinsic_Tmatrix = self.extrinsic_Tmatrix.reshape(3,1)
        # print("extrinsic_Tmatrix", self.extrinsic_Tmatrix)

        Matrix_calib = np.eye(4)
        Matrix_calib[:3, :3] = self.extrinsic_Rmatrix
        Matrix_calib[:3, 3] = self.extrinsic_Tmatrix.flatten()
        # print("Matrix_calib", Matrix_calib)
        Matrix_T = np.array([[X],[Y],[Z],[1]]).reshape(4,1)
        # print("Matrix_T", Matrix_T)
        if self.object_flag == 0:
            # self.current_class = self.animal_class
            if int(Matrix_T[0][0]) != 0:
                self.length = parameter[6]
                self.Matrix_pos = np.dot(Matrix_calib, Matrix_T) # vi tri XYZ
                self.xpos = 0
            else:
                self.xpos = 1

        X_pos = float(self.Matrix_pos[0][0])
        Y_pos = float(self.Matrix_pos[1][0])
        Z_pos = float(self.Matrix_pos[2][0])
        print("X_pos", X_pos)
        print("Y_pos", Y_pos)
        print("Z_pos", Z_pos)

        if 0 < int(Yaw) <= 180 or -180 <= int(Yaw) < 0:
            Yaw = Yaw
            # print("yaw1", Yaw)
        elif 180 < int(Yaw) <= 360:
            Yaw = -180 + (Yaw - 180)
            # print("yaw2", Yaw)
        else:
            Yaw = -(180 + (Yaw + 180))
            # print('yaw3', Yaw)

        if self.object_flag == 0:
            if int(Matrix_T[0][0]) != 0:
                self.Matrix_angle = np.array([[Roll], [Pitch], [Yaw]]) # goc pitch, yaw
        Roll_pos = float(self.Matrix_angle[0][0])
        Pitch_pos = float(self.Matrix_angle[1][0])
        Yaw_pos = float(self.Matrix_angle[2][0])
        print("Roll_pos", Roll_pos)
        print("Pitch_pos", Pitch_pos)
        print("Yaw_pos", Yaw_pos)

        if self.animal_class == "seal":
            # vi tri XYZ
            if X_pos == X_pos and self.xpos == 1:
                delta = np.array([0, 0, 0, 0])
            elif 180 < X_pos <= 190:
                delta = np.array([50, -60, 0, 0])
                if -50 < Y_pos <= -40:
                    delta = np.array([-50, 175, 20, 0])
            elif 220 < X_pos <= 230:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-50, 175, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 175, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-10, 165, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-30, 165, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-30, 160, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-50, 175, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-30, 175, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-50, 175, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-30, 175, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-50, 175, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 175, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 175, 20, 0])
            elif 230 < X_pos <= 240:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-50, 175, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 175, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-10, 165, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-50, 165, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-50, 160, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-50, 175, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-30, 175, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-50, 165, 20, 0]) #[..,155]
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-40, 165, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 165, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-40, 165, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 175, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 175, 20, 0])
            elif 240 < X_pos <= 250:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-10, 165, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 165, 20, 0]) #done
                if -220 < Y_pos <= -210:
                    delta = np.array([-10, 161, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-20, 161, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-20, 160, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-50, 160, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-30, 166, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-50, 150, 20, 0]) #[..,160..]
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-50, 156, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 165, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 165, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 165, 20, 0])
            elif 250 < X_pos <= 260:
                delta = np.array([-50, 155, 0, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-50, 160, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-30, 160, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-20, 160, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-20, 160, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-20, 140, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-40, 140, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-20, 145, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-40, 145, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-50, 150, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 150, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 140, 20, 0]) #done
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 140, 20, 0]) #done
                if -310 < Y_pos <= -300:
                    delta = np.array([-44, 140, 20, 0]) #done
            elif 260 < X_pos <= 270:
                delta = np.array([-50, 150, 0, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-12, 148, 20, 0]) #done [-50,...]
                if -220 < Y_pos <= -210:
                    delta = np.array([-12, 148, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-17, 150, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-30, 140, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-30, 150, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-35, 140, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-35, 140, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-35, 140, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-70, 140, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-35, 120, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-70, 120, 20, 0])  # done
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 120, 20, 0]) #done
                if -310 < Y_pos <= -300:
                    delta = np.array([-50, 130, 20, 0]) #done
            elif 270 < X_pos <= 280:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-10, 150, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 150, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-20, 150, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-30, 150, 20, 0])
                if -240 < Y_pos <= -230:
                    delta = np.array([-30, 150, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-30, 150, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-50, 130, 20, 0])  # done
                if -260 < Y_pos <= -250:
                    delta = np.array([-30, 130, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-50, 130, 20, 0])  # done
                if -270 < Y_pos <= -260:
                    delta = np.array([-30, 135, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 130, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 130, 20, 0])
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 120, 20, 0]) #done
                if -310 < Y_pos <= -300:
                    delta = np.array([-50, 130, 20, 0]) #done
            elif 280 < X_pos <= 290:
                delta = np.array([-50, 100, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-10, 120, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-15, 120, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-20, 150, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-35, 145, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-22, 120, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-28, 120, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-44, 120, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-35, 125, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-44, 125, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-35, 120, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-40, 100, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 100, 20, 0])
            elif 290 < X_pos <= 300:
                delta = np.array([-40, 120, 0, 0])
                if -190 < Y_pos <= -180:
                    delta = np.array([-10, 135, 20, 0]) #done
                if -200 < Y_pos <= -190:
                    delta = np.array([-10, 135, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-20, 120, 20, 0]) #done
                if -220 < Y_pos <= -210:
                    delta = np.array([-40, 145, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-28, 140, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-40, 130, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-30, 130, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-54, 130, 20, 0]) #done
                    if -110 < Yaw_pos < -50:
                        delta = np.array([-30, 130, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-40, 105, 20, 0])
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-54, 105, 20, 0]) #done
                    if -110 < Yaw_pos < -50:
                        delta = np.array([-30, 130, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 120, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-40, 120, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 115, 20, 0]) #done
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 120, 20, 0]) #done
                if -310 < Y_pos <= -300:
                    delta = np.array([-50, 130, 20, 0]) #done
            elif 300 < X_pos <= 310:
                delta = np.array([-20, 127, 20, 0])
                if -190 < Y_pos <= -180:
                    delta = np.array([-20, 127, 20, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-20, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-20, 127, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-16, 127, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-20, 127, 20, 0])
                if -240 < Y_pos <= -230:
                    delta = np.array([-20, 130, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-20, 114, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-30, 114, 20, 0]) #done
                    if -110 < Yaw_pos < -50:
                        delta = np.array([-30, 130, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-43, 120, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-30, 114, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-20, 130, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-20, 130, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-40, 100, 20, 0])
            elif 310 < X_pos <= 320:
                delta = np.array([-30, 127, 20, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-30, 100, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-30, 100, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-40, 125, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-40, 120, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-30, 133, 20, 0]) #done
                    if -100 < Yaw_pos < -50:
                        delta = np.array([-30, 133, 20, 0])  # done
                if -250 < Y_pos <= -240:
                    delta = np.array([-40, 120, 20, 0]) # done
                    if -190 < Yaw_pos < -150:
                        delta = np.array([-40, 110, 20, 0])  # done
                if -260 < Y_pos <= -250:
                    delta = np.array([-35, 120, 20, 0]) #done
                    if -190 < Yaw_pos < -160:
                        delta = np.array([-44, 110, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-50, 102, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 102, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-30, 100, 20, 0])
            elif 320 < X_pos <= 330:
                delta = np.array([-40, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-40, 120, 20, 0]) #done
                if -220 < Y_pos <= -210:
                    delta = np.array([-40, 120, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-40, 110, 20, 0])
                if -240 < Y_pos <= -230:
                    delta = np.array([-40, 110, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-40, 120, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-32, 105, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 110, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-30, 100, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-40, 100, 20, 0])
            elif 330 < X_pos <= 340:
                delta = np.array([-57, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-57, 100, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-57, 100, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-57, 120, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-57, 105, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-57, 80, 20, 0])
                if -260 < Y_pos <= -250:
                    delta = np.array([-57, 80, 20, 0])
                if -270 < Y_pos <= -260:
                    delta = np.array([-43, 73, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-43, 73, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-57, 80, 20, 0])
            elif 340 < X_pos <= 350:
                delta = np.array([-57, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-57, 90, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-57, 90, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-57, 90, 20, 0])
                if -240 < Y_pos <= -230:
                    delta = np.array([-57, 90, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-57, 98, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-67, 90, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-67, 90, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-57, 80, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-57, 80, 20, 0])
            else:
                delta = np.array([-50, 130, 50, 0])  # mm
            # vi tri XYZ
            self.Matrix_pos = self.Matrix_pos + delta.reshape(-1, 1)
            while not 180 <= float(self.Matrix_pos[0][0]) <= 295: # X
                if float(self.Matrix_pos[0][0]) > 295:
                    self.Matrix_pos[0] -= 10
                if float(self.Matrix_pos[0][0]) < 180:
                    self.Matrix_pos[0] += 10
            while not -180 <= float(self.Matrix_pos[1][0]) <= -40: # Y
                if float(self.Matrix_pos[1][0]) >= -40:
                    self.Matrix_pos[1] -= 20
                if float(self.Matrix_pos[1][0]) <= -180:
                    self.Matrix_pos[1] += 20
            while not -49 <= float(self.Matrix_pos[2][0]) <= -45:
                if float(self.Matrix_pos[2][0]) >= -45:
                    self.Matrix_pos[2] -= 4
                if float(self.Matrix_pos[2][0]) <= -49:
                    self.Matrix_pos[2] += 4
            print("Ma tran vi tri:", self.Matrix_pos)

            # goc pitch, yaw
            if X_pos == X_pos and self.xpos == 1:
                delta_angle = np.array([0, 0, 0])
            elif 0 < Roll_pos < 90:
                delta_roll = 90 - Roll_pos
                self.Matrix_angle[0] = 180 - delta_roll
                delta_angle = np.array([0, 0, 60])
                if -30 <= Pitch_pos <= -20:
                    delta_angle = np.array([0, 10, 60])
                if -40 <= Pitch_pos <= -30:
                    delta_angle = np.array([0, 20, 60])
                print("3")
            elif -145 < Roll_pos < 145:
                delta_angle = np.array([40, 0, 0])
                if 0 < Roll_pos < 145: # duong
                    delta_angle = np.array([40, 0, 0])
                    if -50 < Yaw_pos <= -40:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([40, 0, 10]) #done
                    if -60 < Yaw_pos <= -50:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([40, 0, 10]) #done
                    if -70 < Yaw_pos <= -60:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([40, 0, 10]) # done
                    if -80 < Yaw_pos <= -70:
                        delta_angle = np.array([40, 0, 77])
                    if -90 < Yaw_pos <= -80:
                        delta_angle = np.array([40, 0, 87])
                    if -100 < Yaw_pos <= -90:
                        delta_angle = np.array([40, 0, 95])
                    if -110 < Yaw_pos <= -100:
                        delta_angle = np.array([40, 0, 95])
                    if -130 < Yaw_pos <= -110:
                        self.Matrix_angle[2] = 0
                        delta_angle = np.array([-50, 0, 0])
                    if -140 < Yaw_pos <= -130:
                        delta_angle = np.array([90, 0, 180])
                    if -160 < Yaw_pos <= -140:
                        delta_angle = np.array([90, 0, 90])
                    if -180 < Yaw_pos <= -160:
                        delta_angle = np.array([320, 0, 110])  # done
                    if -200 < Yaw_pos <= -180:
                        delta_angle = np.array([340, 0, 140])  # done
                    if Yaw_pos <= -200:
                        delta_angle = np.array([90, 0, 150])
                elif -70 < Roll_pos < 0: # -4x
                    delta_angle = np.array([-135, 0, 0])
                    if -50 < Yaw_pos <= -40:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-135, 0, 47])
                    if -60 < Yaw_pos <= -50:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-135, 0, 57])
                    if -70 < Yaw_pos <= -60:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-135, 0, 67])
                    if -80 < Yaw_pos <= -70:
                        delta_angle = np.array([-135, 0, 77])
                    if -90 < Yaw_pos <= -80:
                        delta_angle = np.array([-135, 0, 87])
                    if -100 < Yaw_pos <= -90:
                        delta_angle = np.array([-135, 0, 95])
                    if -110 < Yaw_pos <= -100:
                        delta_angle = np.array([-135, 0, 95])
                    if -130 < Yaw_pos <= -110:
                        self.Matrix_angle[2] = 0
                        delta_angle = np.array([-50, 0, 0])
                    if -140 < Yaw_pos <= -130:
                        delta_angle = np.array([320, 0, 180])
                    if -160 < Yaw_pos <= -140:
                        self.Matrix_angle[1] = 0
                        delta_angle = np.array([320, 0, 190])
                    if -180 < Yaw_pos <= -160:
                        delta_angle = np.array([340, 0, 110])  # done
                    if -200 < Yaw_pos <= -180:
                        delta_angle = np.array([240, 0, 140])  # done
                    if Yaw_pos <= -200:
                        delta_angle = np.array([240, 0, 150])
                elif -100 < Roll_pos < -70:
                    delta_angle = np.array([-90, 0, 0])
                    if -50 < Yaw_pos <= -40:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-90, 0, 47])
                    if -60 < Yaw_pos <= -50:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-90, 0, 57])
                    if -70 < Yaw_pos <= -60:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-90, 0, 67])
                    if -80 < Yaw_pos <= -70:
                        delta_angle = np.array([-90, 0, 77])
                    if -90 < Yaw_pos <= -80:
                        delta_angle = np.array([-90, 0, 87])
                    if -100 < Yaw_pos <= -90:
                        delta_angle = np.array([-90, 0, 95])
                    if -110 < Yaw_pos <= -100:
                        delta_angle = np.array([-90, 0, 95])
                    if -130 < Yaw_pos <= -110:
                        self.Matrix_angle[2] = 0
                        delta_angle = np.array([-50, 0, 0])
                    if -140 < Yaw_pos <= -130:
                        delta_angle = np.array([-90, 0, 180])
                    if -160 < Yaw_pos <= -140:
                        delta_angle = np.array([-90, 0, 90])
                    if -180 < Yaw_pos <= -160:
                        delta_angle = np.array([-90, 0, 110]) # done
                    if -200 < Yaw_pos <= -180:
                        delta_angle = np.array([240, 0, 140]) # done
                    if Yaw_pos <= -200:
                        delta_angle = np.array([240, 0, 150])
                elif -145 < Roll_pos < -100:
                    delta_angle = np.array([-80, 0, 0]) # [-140,-100]
                    if -50 < Yaw_pos <= -40:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-80, 0, 47])
                    if -60 < Yaw_pos <= -50:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-80, 0, 57])
                    if -70 < Yaw_pos <= -60:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-80, 0, 67])
                    if -80 < Yaw_pos <= -70:
                        delta_angle = np.array([-80, 0, 77])
                    if -90 < Yaw_pos <= -80:
                        delta_angle = np.array([-80, 0, 87])
                    if -100 < Yaw_pos <= -90:
                        delta_angle = np.array([-80, 0, 95])
                    if -110 < Yaw_pos <= -100:
                        delta_angle = np.array([-80, 0, 95])
                    if -130 < Yaw_pos <= -110:
                        self.Matrix_angle[2] = 0
                        delta_angle = np.array([-50, 0, 0])
                    if -140 < Yaw_pos <= -130:
                        delta_angle = np.array([-90, 0, 180])
                    if -160 < Yaw_pos <= -140:
                        delta_angle = np.array([320, 0, 190]) #done
                    if -180 < Yaw_pos <= -160:
                        delta_angle = np.array([340, 0, 110]) # done
                    if -200 < Yaw_pos <= -180:
                        delta_angle = np.array([240, 0, 140]) # done
                    if Yaw_pos <= -200:
                        delta_angle = np.array([240, 0, 150])
                print("4")
            elif Roll_pos <= -145 or 145 <= Roll_pos:
                delta_angle = np.array([0, 0, 60])
                if 145 <= Roll_pos:
                    if -50 < Yaw_pos <= -40:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([10, 0, 47])
                    if -60 < Yaw_pos <= -50:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([10, 0, 57])
                    if -70 < Yaw_pos <= -60:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([10, 0, 67])
                    if -80 < Yaw_pos <= -70:
                        delta_angle = np.array([10, 0, 60])
                        if 280 < X_pos <= 290:
                            delta_angle = np.array([10, 0, 60])
                    if -90 < Yaw_pos <= -80:
                        delta_angle = np.array([10, 0, 87])
                    if -100 < Yaw_pos <= -90:
                        delta_angle = np.array([10, 0, 95])
                    if -110 < Yaw_pos <= -100:
                        delta_angle = np.array([0, 0, 95])
                    if -130 < Yaw_pos <= -110:
                        self.Matrix_angle[2] = 0
                        delta_angle = np.array([10, 0, 0])
                    if -140 < Yaw_pos <= -130:
                        delta_angle = np.array([10, 0, 180])
                    if -160 < Yaw_pos <= -140:
                        delta_angle = np.array([320, 0, 90])
                    if -180 < Yaw_pos <= -160:
                        delta_angle = np.array([340, 0, 110])  # done
                    if -200 < Yaw_pos <= -180:
                        delta_angle = np.array([240, 0, 140])  # done
                    if Yaw_pos <= -200:
                        delta_angle = np.array([240, 0, 150])
                else:
                    if -50 < Yaw_pos <= -40:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-10, 0, 47])
                    if -60 < Yaw_pos <= -50:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-10, 0, 57])
                    if -70 < Yaw_pos <= -60:
                        self.Matrix_angle[1] = -2
                        delta_angle = np.array([-10, 0, 67])
                    if -80 < Yaw_pos <= -70:
                        delta_angle = np.array([-10, 0, 77])
                    if -90 < Yaw_pos <= -80:
                        delta_angle = np.array([-10, 0, 87])
                    if -100 < Yaw_pos <= -90:
                        delta_angle = np.array([-10, 0, 95])
                    if -110 < Yaw_pos <= -100:
                        delta_angle = np.array([0, 0, 95])
                    if -130 < Yaw_pos <= -110:
                        self.Matrix_angle[2] = 0
                        delta_angle = np.array([-50, 0, 0])
                    if -140 < Yaw_pos <= -130:
                        delta_angle = np.array([-90, 0, 180])
                    if -160 < Yaw_pos <= -140:
                        delta_angle = np.array([320, 0, 90])
                    if -180 < Yaw_pos <= -160:
                        delta_angle = np.array([340, 0, 110])  # done
                    if -200 < Yaw_pos <= -180:
                        delta_angle = np.array([240, 0, 140])  # done
                    if Yaw_pos <= -200:
                        delta_angle = np.array([240, 0, 150])
                print("5")
            else:
                delta_angle = np.array([0, 0, 60])
                print("6")
            self.Matrix_angle = self.Matrix_angle + delta_angle.reshape(-1, 1)

            while not -200 <= float(self.Matrix_angle[0][0]) <= 200: # -180 < roll < -140 and 140 < roll < 180
                if float(self.Matrix_angle[0][0]) > 200:
                    self.Matrix_angle[0] -= 2
                if float(self.Matrix_angle[0][0]) < -200:
                    self.Matrix_angle[0] += 2
            while not -10 <= float(self.Matrix_angle[1][0]) <= 10: # pitch
                if float(self.Matrix_angle[1][0]) > 10:
                    self.Matrix_angle[1][0] -= 2
                if float(self.Matrix_angle[1][0]) < -10:
                    self.Matrix_angle[1][0] += 2
            while not -90 <= float(self.Matrix_angle[2][0]) <= 90: # yaw
                if float(self.Matrix_angle[2][0]) > 90:
                    self.Matrix_angle[2][0] -= 4
                if float(self.Matrix_angle[2][0]) < -90:
                    self.Matrix_angle[2][0] += 4
            if Y_pos < -250:
                if self.Matrix_angle[0][0] > 0:
                    self.Matrix_angle[0][0] = -self.Matrix_angle[0][0]
                    print(self.Matrix_angle[0][0])
            else:
                if self.Matrix_angle[0][0] < 0:
                    self.Matrix_angle[0][0] = abs(self.Matrix_angle[0][0])
            print("Ma tran goc xoay:", self.Matrix_angle)

        if self.animal_class == "duck":
             # vi tri XYZ
            if X_pos == X_pos and self.xpos == 1:
                delta = np.array([0, 0, 0, 0])
            elif 220 < X_pos <= 230:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-50, 175, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 175, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-17, 172, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-17, 172, 20, 0])  # done
                if -240 < Y_pos <= -230:
                    delta = np.array([-17, 172, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-50, 175, 20, 0])
                if -260 < Y_pos <= -250:
                    delta = np.array([-50, 175, 20, 0])
                if -270 < Y_pos <= -260:
                    delta = np.array([-50, 175, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 175, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 175, 20, 0])
            elif 230 < X_pos <= 240:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-50, 175, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 175, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-10, 165, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-25, 165, 20, 0])  # done
                if -240 < Y_pos <= -230:
                    delta = np.array([-28, 166, 20, 0])  # done
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 125, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-29, 164, 20, 0]) #done
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 125, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-32, 155, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-28, 175, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-28, 175, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 175, 20, 0])
            elif 240 < X_pos <= 250:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-10, 165, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 165, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-10, 165, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-15, 156, 20, 0])  # done
                if -240 < Y_pos <= -230:
                    delta = np.array([-21, 155, 20, 0]) #done
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 125, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-50, 160, 20, 0])  # done
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 145, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-50, 160, 20, 0])
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 125, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-17, 150, 20, 0]) #done [-37,..]
                if -280 < Y_pos <= -270:
                    delta = np.array([-40, 155, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 165, 20, 0])
            elif 250 < X_pos <= 260:
                delta = np.array([-50, 155, 0, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 158, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-30, 160, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-20, 160, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-25, 147, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-25, 145, 20, 0]) #done
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 175, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-20, 145, 20, 0])  # done
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 175, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-30, 140, 20, 0]) #done [-36,...]
                    if -190 < Yaw_pos < -100:
                        delta = np.array([-35, 175, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-30, 150, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 126, 20, 0])  #done
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 130, 20, 0])  # doneR
                if -310 < Y_pos <= -300:
                    delta = np.array([-44, 140, 20, 0]) #done
            elif 260 < X_pos <= 270:
                delta = np.array([-50, 150, 0, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-12, 148, 20, 0])  # done
                if -220 < Y_pos <= -210:
                    delta = np.array([-12, 148, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-17, 150, 20, 0])  #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-30, 140, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-30, 150, 20, 0])  # done
                if -260 < Y_pos <= -250:
                    delta = np.array([-35, 140, 20, 0])
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 130, 20, 0]) #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-42, 130, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-45, 125, 20, 0]) #done
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 123, 20, 0])  #done
                if -310 < Y_pos <= -300:
                    delta = np.array([-50, 130, 20, 0])  # done
            elif 270 < X_pos <= 280:
                delta = np.array([-50, 175, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-10, 150, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-10, 150, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-20, 150, 20, 0])  # done
                if -230 < Y_pos <= -220:
                    delta = np.array([-30, 150, 20, 0])
                if -240 < Y_pos <= -230:
                    delta = np.array([-20, 140, 20, 0]) #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-13, 130, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-30, 135, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-30, 135, 20, 0])  # done
                if -280 < Y_pos <= -270:
                    delta = np.array([-50, 120, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 115, 20, 0]) #done
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 120, 20, 0])  # done
                if -310 < Y_pos <= -300:
                    delta = np.array([-50, 130, 20, 0])  # done
            elif 280 < X_pos <= 290:
                delta = np.array([-50, 100, 0, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-0, 150, 20, 0]) #done
                if -210 < Y_pos <= -200:
                    delta = np.array([-15, 136, 20, 0]) #done
                if -220 < Y_pos <= -210:
                    delta = np.array([-20, 137, 20, 0])  #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-35, 145, 20, 0])  # done
                if -240 < Y_pos <= -230:
                    delta = np.array([-20, 120, 20, 0])  # done
                if -250 < Y_pos <= -240:
                    delta = np.array([-30, 120, 20, 0])
                if -260 < Y_pos <= -250:
                    delta = np.array([-35, 125, 20, 0])  # done
                if -270 < Y_pos <= -260:
                    delta = np.array([-35, 125, 20, 0])  #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-40, 120, 20, 0]) #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-40, 110, 20, 0])
            elif 290 < X_pos <= 300:
                delta = np.array([-40, 120, 0, 0])
                if -190 < Y_pos <= -180:
                    delta = np.array([-10, 135, 20, 0]) #done
                if -200 < Y_pos <= -190:
                    delta = np.array([-10, 137, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-20, 145, 20, 0]) #done
                if -220 < Y_pos <= -210:
                    delta = np.array([-40, 145, 20, 0])  # done
                if -230 < Y_pos <= -220:
                    delta = np.array([-30, 140, 20, 0])  # done
                if -240 < Y_pos <= -230:
                    delta = np.array([-40, 130, 20, 0])  # done
                if -250 < Y_pos <= -240:
                    delta = np.array([-30, 130, 20, 0])  # done
                if -260 < Y_pos <= -250:
                    delta = np.array([-40, 105, 20, 0])
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 120, 20, 0])  # done
                if -280 < Y_pos <= -270:
                    delta = np.array([-40, 120, 20, 0])  # done
                if -290 < Y_pos <= -280:
                    delta = np.array([-50, 110, 20, 0])  #done
                if -300 < Y_pos <= -290:
                    delta = np.array([-50, 120, 20, 0])  # done
                if -310 < Y_pos <= -300:
                    delta = np.array([-50, 130, 20, 0])  # done
            elif 300 < X_pos <= 310:
                delta = np.array([-20, 127, 20, 0])
                if -190 < Y_pos <= -180:
                    delta = np.array([-11, 137, 20, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-20, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-17, 127, 20, 0]) #done
                if -220 < Y_pos <= -210:
                    delta = np.array([-20, 127, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-23, 120, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-30, 126, 20, 0])  #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-30, 124, 20, 0])  #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-40, 110, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 110, 20, 0])  #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-40, 105, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-20, 100, 20, 0])
            elif 310 < X_pos <= 320:
                delta = np.array([-30, 127, 20, 0])
                if -200 < Y_pos <= -190:
                    delta = np.array([-30, 125, 20, 0]) #done
                if -210 < Y_pos <= -200:
                    delta = np.array([-30, 100, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-40, 125, 20, 0])  # done
                if -230 < Y_pos <= -220:
                    delta = np.array([-20, 130, 20, 0])  #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-30, 120, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-40, 120, 20, 0])  # done
                if -260 < Y_pos <= -250:
                    delta = np.array([-35, 110, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-50, 102, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-45, 102, 20, 0])  #done
                if -290 < Y_pos <= -280:
                    delta = np.array([-30, 100, 20, 0])
            elif 320 < X_pos <= 330:
                delta = np.array([-40, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-30, 127, 20, 0])  # done
                if -210 < Y_pos <= -200:
                    delta = np.array([-25, 123, 20, 0])  # done
                if -220 < Y_pos <= -210:
                    delta = np.array([-30, 119, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-35, 116, 20, 0]) #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-40, 110, 20, 0])  # done
                if -250 < Y_pos <= -240:
                    delta = np.array([-34, 110, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-40, 105, 20, 0])  #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-40, 106, 20, 0])  #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-30, 100, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-40, 100, 20, 0])
            elif 330 < X_pos <= 340:
                delta = np.array([-57, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-57, 100, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-42, 124, 20, 0]) #done
                if -230 < Y_pos <= -220:
                    delta = np.array([-37, 110, 20, 0])  #done
                if -240 < Y_pos <= -230:
                    delta = np.array([-47, 110, 20, 0])  #done
                if -250 < Y_pos <= -240:
                    delta = np.array([-43, 105, 20, 0]) #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-40, 105, 20, 0]) #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-48, 88, 20, 0])  #done
                if -280 < Y_pos <= -270:
                    delta = np.array([-43, 73, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-57, 80, 20, 0])
            elif 340 < X_pos <= 350:
                delta = np.array([-57, 127, 20, 0])
                if -210 < Y_pos <= -200:
                    delta = np.array([-57, 90, 20, 0])
                if -220 < Y_pos <= -210:
                    delta = np.array([-57, 90, 20, 0])
                if -230 < Y_pos <= -220:
                    delta = np.array([-57, 90, 20, 0])
                if -240 < Y_pos <= -230:
                    delta = np.array([-57, 90, 20, 0])
                if -250 < Y_pos <= -240:
                    delta = np.array([-44, 98, 20, 0])  #done
                if -260 < Y_pos <= -250:
                    delta = np.array([-57, 85, 20, 0])  #done
                if -270 < Y_pos <= -260:
                    delta = np.array([-67, 90, 20, 0])
                if -280 < Y_pos <= -270:
                    delta = np.array([-57, 80, 20, 0])
                if -290 < Y_pos <= -280:
                    delta = np.array([-57, 80, 20, 0])
            else:
                delta = np.array([-50, 130, 50, 0])  # mm
            # elif 230 < X_pos <= 240:
            #     delta = np.array([-50, 175, 0, 0])
            #     if -200 < Y_pos <= -190:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-50, 175, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-50, 175, 20, 0])
            # elif 240 < X_pos <= 250:
            #     delta = np.array([-50, 165, 0, 0])
            #     if -200 < Y_pos <= -190:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-50, 165, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-50, 165, 20, 0])
            # elif 250 < X_pos <= 260:
            #     delta = np.array([-50, 155, 0, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-50, 150, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-50, 150, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-50, 150, 20, 0])
            # elif 260 < X_pos <= 270:
            #     delta = np.array([-50, 150, 0, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-50, 160, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-50, 140, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-50, 140, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-50, 140, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-50, 140, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-50, 140, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-50, 140, 20, 0])
            # elif 270 < X_pos <= 280:
            #     delta = np.array([-50, 130, 0, 0])
            #     if -200 < Y_pos <= -190:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-50, 130, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-50, 130, 20, 0])
            # elif 280 < X_pos <= 290:
            #     delta = np.array([-50, 100, 0, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-50, 120, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-50, 120, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-50, 120, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-50, 120, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-50, 120, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-50, 100, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-50, 100, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-50, 100, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-50, 100, 20, 0])
            # elif 290 < X_pos <= 300:
            #     delta = np.array([-40, 120, 0, 0])
            #     if -200 < Y_pos <= -190:
            #         delta = np.array([-10, 135, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-40, 120, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-40, 120, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-40, 110, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-40, 110, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-40, 110, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-40, 105, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-40, 105, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-40, 105, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-40, 105, 20, 0])
            # elif 300 < X_pos <= 310:
            #     delta = np.array([-20, 127, 20, 0])
            #     if -190 < Y_pos <= -180:
            #         delta = np.array([-20, 127, 20, 0])
            #     if -200 < Y_pos <= -190:
            #         delta = np.array([-20, 127, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-20, 127, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-20, 127, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-20, 127, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-20, 100, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-20, 100, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-50, 100, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-50, 100, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-50, 100, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-20, 100, 20, 0])
            # elif 310 < X_pos <= 320:
            #     delta = np.array([-30, 127, 20, 0])
            #     if -200 < Y_pos <= -190:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-30, 100, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-30, 100, 20, 0])
            # elif 320 < X_pos <= 330:
            #     delta = np.array([-40, 127, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-40, 110, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-40, 110, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-40, 110, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-40, 100, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-40, 100, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-40, 100, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-40, 100, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-40, 100, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-40, 100, 20, 0])
            # elif 330 < X_pos <= 340:
            #     delta = np.array([-57, 127, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-57, 100, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-57, 100, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-57, 100, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-57, 80, 20, 0])
            # elif 340 < X_pos <= 350:
            #     delta = np.array([-57, 127, 20, 0])
            #     if -210 < Y_pos <= -200:
            #         delta = np.array([-57, 90, 20, 0])
            #     if -220 < Y_pos <= -210:
            #         delta = np.array([-57, 90, 20, 0])
            #     if -230 < Y_pos <= -220:
            #         delta = np.array([-57, 90, 20, 0])
            #     if -240 < Y_pos <= -230:
            #         delta = np.array([-57, 90, 20, 0])
            #     if -250 < Y_pos <= -240:
            #         delta = np.array([-57, 90, 20, 0])
            #     if -260 < Y_pos <= -250:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -270 < Y_pos <= -260:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -280 < Y_pos <= -270:
            #         delta = np.array([-57, 80, 20, 0])
            #     if -290 < Y_pos <= -280:
            #         delta = np.array([-57, 80, 20, 0])
            # else:
            #     delta = np.array([-50, 130, 50, 0])  # mm
            # vi tri XYZ
            self.Matrix_pos = self.Matrix_pos + delta.reshape(-1, 1)
            while not 180 <= float(self.Matrix_pos[0][0]) <= 295: # X
                if float(self.Matrix_pos[0][0]) > 295:
                    self.Matrix_pos[0] -= 10
                if float(self.Matrix_pos[0][0]) < 180:
                    self.Matrix_pos[0] += 10
            while not -180 <= float(self.Matrix_pos[1][0]) <= -40: # Y
                if float(self.Matrix_pos[1][0]) >= -40:
                    self.Matrix_pos[1] -= 20
                if float(self.Matrix_pos[1][0]) <= -180:
                    self.Matrix_pos[1] += 20
            while not -49 <= float(self.Matrix_pos[2][0]) <= -45:
                if float(self.Matrix_pos[2][0]) >= -45:
                    self.Matrix_pos[2] -= 4
                if float(self.Matrix_pos[2][0]) <= -49:
                    self.Matrix_pos[2] += 4
            print("Ma tran vi tri:", self.Matrix_pos)

            # # goc pitch, yaw
            # if 0 < Roll_pos < 90:
            #     delta_roll = 90 - Roll_pos
            #     self.Matrix_angle[0] = 180 - delta_roll
            #     delta_angle = np.array([0, 0, 60])
            #     if -30 <= Pitch_pos <= -20:
            #         delta_angle = np.array([0, 10, 60])
            #     if -40 <= Pitch_pos <= -30:
            #         delta_angle = np.array([0, 20, 60])
            #     print("3")
            # elif -145 < Roll_pos < 145:
            #     if 0 < Roll_pos < 140:
            #         delta_angle = np.array([80, 0, 0])
            #         if -50 < Yaw_pos <= -40:
            #             delta_angle = np.array([80, 0, 47])
            #         if -60 < Yaw_pos <= -50:
            #             delta_angle = np.array([80, 0, 57])
            #         if -70 < Yaw_pos <= -60:
            #             delta_angle = np.array([80, 0, 67])
            #         if -80 < Yaw_pos <= -70:
            #             delta_angle = np.array([80, 0, 77])
            #         if -90 < Yaw_pos <= -80:
            #             delta_angle = np.array([80, 0, 87])
            #         if -100 < Yaw_pos <= -90:
            #             delta_angle = np.array([80, 0, 95])
            #         if -110 < Yaw_pos <= -100:
            #             delta_angle = np.array([80, 0, 95])
            #         if Yaw_pos <= -110:
            #             self.Matrix_angle[2] = 0
            #     elif -70 < Roll_pos < 0: # -4x
            #         delta_angle = np.array([-135, 0, 0])
            #         if -50 < Yaw_pos <= -40:
            #             delta_angle = np.array([-135, 0, 47])
            #         if -60 < Yaw_pos <= -50:
            #             delta_angle = np.array([-135, 0, 57])
            #         if -70 < Yaw_pos <= -60:
            #             delta_angle = np.array([-135, 0, 67])
            #         if -80 < Yaw_pos <= -70:
            #             delta_angle = np.array([-135, 0, 77])
            #         if -90 < Yaw_pos <= -80:
            #             delta_angle = np.array([-135, 0, 87])
            #         if -100 < Yaw_pos <= -90:
            #             delta_angle = np.array([-135, 0, 95])
            #         if -110 < Yaw_pos <= -100:
            #             delta_angle = np.array([-135, 0, 95])
            #         if Yaw_pos <= -110:
            #             self.Matrix_angle[2] = 0
            #     elif -100 < Roll_pos < -70:
            #         delta_angle = np.array([-90, 0, 0])
            #         if -50 < Yaw_pos <= -40:
            #             delta_angle = np.array([-90, 0, 47])
            #         if -60 < Yaw_pos <= -50:
            #             delta_angle = np.array([-90, 0, 57])
            #         if -70 < Yaw_pos <= -60:
            #             delta_angle = np.array([-90, 0, 67])
            #         if -80 < Yaw_pos <= -70:
            #             delta_angle = np.array([-90, 0, 77])
            #         if -90 < Yaw_pos <= -80:
            #             delta_angle = np.array([-90, 0, 87])
            #         if -100 < Yaw_pos <= -90:
            #             delta_angle = np.array([-90, 0, 95])
            #         if -110 < Yaw_pos <= -100:
            #             delta_angle = np.array([-90, 0, 95])
            #         if Yaw_pos <= -110:
            #             self.Matrix_angle[2] = 0
            #     else:
            #         delta_angle = np.array([-80, 0, 0]) # [-140,-100]
            #         if -50 < Yaw_pos <= -40:
            #             delta_angle = np.array([-80, 0, 47])
            #         if -60 < Yaw_pos <= -50:
            #             delta_angle = np.array([-80, 0, 57])
            #         if -70 < Yaw_pos <= -60:
            #             delta_angle = np.array([-80, 0, 67])
            #         if -80 < Yaw_pos <= -70:
            #             delta_angle = np.array([-80, 0, 77])
            #         if -90 < Yaw_pos <= -80:
            #             delta_angle = np.array([-80, 0, 87])
            #         if -100 < Yaw_pos <= -90:
            #             delta_angle = np.array([-80, 0, 95])
            #         if -110 < Yaw_pos <= -100:
            #             delta_angle = np.array([-80, 0, 95])
            #         if Yaw_pos <= -110:
            #             self.Matrix_angle[2] = 0
            #     print("4")
            # elif Roll_pos <= -145 or 145 <= Roll_pos:
            #     delta_angle = np.array([0, 0, 60])
            #     if -50 < Yaw_pos <= -40:
            #         delta_angle = np.array([0, 0, 47])
            #     if -60 < Yaw_pos <= -50:
            #         delta_angle = np.array([0, 0, 57])
            #     if -70 < Yaw_pos <= -60:
            #         delta_angle = np.array([0, 0, 67])
            #     if -80 < Yaw_pos <= -70:
            #         delta_angle = np.array([0, 0, 77])
            #     if -90 < Yaw_pos <= -80:
            #         delta_angle = np.array([0, 0, 87])
            #     if -100 < Yaw_pos <= -90:
            #         delta_angle = np.array([0, 0, 95])
            #     if -110 < Yaw_pos <= -100:
            #         delta_angle = np.array([0, 0, 95])
            #     if Yaw_pos <= -110:
            #         self.Matrix_angle[2] = 0
            #         delta_angle = np.array([0, 0, 0])
            #     print("5")
            # else:
            #     delta_angle = np.array([0, 0, 60])
            #     print("6")
            # self.Matrix_angle = self.Matrix_angle + delta_angle.reshape(-1, 1)
            #
            # # while -140 <= float(self.Matrix_angle[0][0]) <= 140: # -180 < roll < -140 and 140 < roll < 180
            # #     if 0.001 <= float(self.Matrix_angle[0][0]) < 140:
            # #         self.Matrix_angle[0] += 2
            # #     if -140 < float(self.Matrix_angle[0][0]) <= -0.001:
            # #         self.Matrix_angle[0] -= 2
            # while not -10 <= float(self.Matrix_angle[1][0]) <= 10: # pitch
            #     if float(self.Matrix_angle[1][0]) > 10:
            #         self.Matrix_angle[1][0] -= 2
            #     if float(self.Matrix_angle[1][0]) < -10:
            #         self.Matrix_angle[1][0] += 2
            # while not -90 <= float(self.Matrix_angle[2][0]) <= 90: # yaw
            #     if float(self.Matrix_angle[2][0]) > 90:
            #         self.Matrix_angle[2][0] -= 5
            #     if float(self.Matrix_angle[2][0]) < -90:
            #         self.Matrix_angle[2][0] += 5
            # print("Ma tran goc xoay:", self.Matrix_angle)
            # goc roll, pitch, yaw duck
            if X_pos == X_pos and self.xpos == 1:
                delta_angle = np.array([0, 0, 0])
            elif Pitch_pos > 19:
                delta_angle = np.array([0, 0, 0])
                if Roll_pos <= -150 or 150 <= Roll_pos:
                    delta_angle = np.array([0, 0, 0])
                    print("6")
                if 70 <= Pitch_pos <= 80:
                    delta_angle = np.array([0, -60, 0])
                self.Matrix_angle[1] = - float(self.Matrix_angle[1][0])
                print("3+")
            elif -80 <= Pitch_pos <= -70 and Yaw_pos < 40:
                delta_angle = np.angle([0, 66, 0])
                print("1+")
            elif -160 < Roll_pos < -90:
                delta_roll = -90 - Roll_pos
                self.Matrix_angle[0] = 180 - delta_roll
                delta_angle = np.array([0, 0, 0])
                print("4")
                if -30 <= Pitch_pos <= -20:
                    delta_angle = np.array([0, 10, 0])
                if -40 <= Pitch_pos <= -30:
                    delta_angle = np.array([0, 20, 0])
            elif Roll_pos <= -160 or 160 <= Roll_pos:
                delta_angle = np.array([0, 0, 0])
                print("5")
                if -30 <= Pitch_pos <= -20:
                    delta_angle = np.array([0, 10, 0])
                if -40 <= Pitch_pos <= -30:
                    delta_angle = np.array([0, 20, 0])
            elif -50 < Roll_pos < -40:
                delta_angle = np.array([-125, 0, 0])
            elif -60 < Roll_pos < -50:
                delta_angle = np.array([-115, 0, 0])
            elif -70 < Roll_pos < -60:
                delta_angle = np.array([-105, 0, 0])
            elif -80 < Roll_pos < -70:
                delta_angle = np.array([-95, 0, 0])
            else:
                self.Matrix_angle[1] = 1
                delta_angle = np.array([-93, 0, 0]) # [0, 0, 90]
                print("2+")
            self.Matrix_angle = self.Matrix_angle + delta_angle.reshape(-1, 1)
            # roll > 160 and roll < -160
            while -160 <= float(self.Matrix_angle[0][0]) <= 160:
                if 0 <= float(self.Matrix_angle[0][0]) < 160:
                    self.Matrix_angle[0] += 5
                if -160 < float(self.Matrix_angle[0][0]) <= 0:
                    self.Matrix_angle[0] -= 5
            while not -10 <= float(self.Matrix_angle[1][0]) <= 10: # pitch
                if float(self.Matrix_angle[1][0]) > 10:
                    self.Matrix_angle[1][0] -= 2
                if float(self.Matrix_angle[1][0]) < -10:
                    self.Matrix_angle[1][0] += 2
            if 280 < self.Matrix_pos[0]:
                self.Matrix_angle[1] = 1
            if Y_pos < -250:
                if self.Matrix_angle[0] > 0:
                    self.Matrix_angle[0] = -self.Matrix_angle[0]
            self.Matrix_angle[2] = self.interpolation() # noi suy tuyen tinh Yaw
            print("Ma tran goc xoay:", self.Matrix_angle)
        # if self.animal_class == "duck":
        #     # vi tri XYZ 246 < x < 370 -268 < y < -190
        #     if X_pos == X_pos and xpos == 1:
        #         delta = np.array([0, 0, 0, 0])
        #         print("17")
        #     elif 200 < X_pos <= 210:
        #         delta = np.array([0, 0, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-0, 20, 0, 0])
        #     elif 210 < X_pos <= 220:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -170 < Y_pos <= -160:
        #             delta = np.array([-10, 1, 0, 0])
        #     elif 220 < X_pos <= 230:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-20, 140, 0, 0])
        #     elif 230 < X_pos <= 240:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 120, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 120, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 175, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 240 < X_pos <= 250:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-40, 170, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-40, 170, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 250 < X_pos <= 260:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 130, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 130, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 170, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 130, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 130, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 130, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 130, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 145, 0, 0])
        #     elif 260 < X_pos <= 270:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -70 < Y_pos <= -30:
        #             delta = np.array([0, -45, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-5, 170, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-20, 180, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-5, 170, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 270 < X_pos <= 280:
        #         delta = np.array([-30, 170, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-15, 140, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 170, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 180, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 140, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 140, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-35, 168, 0, 0])
        #     elif 280 < X_pos <= 290:
        #         delta = np.array([-40, 170, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-6, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-6, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-6, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-6, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-6, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-6, 130, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-6, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-6, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-40, 170, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-40, 145, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-40, 145, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-50, 168, 0, 0])
        #     elif 290 < X_pos <= 300:
        #         delta = np.array([-10, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([-10, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-10, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-10, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-10, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-10, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-10, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-25, 150, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-10, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-6, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-8, 110, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-6, 145, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-6, 145, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 300 < X_pos <= 310:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 310 < X_pos <= 320:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 144.5, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 144.5, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 144.5, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 90, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 175, 0, 0])
        #     elif 320 < X_pos <= 330:
        #         delta = np.array([-30, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 330 < X_pos <= 340:
        #         delta = np.array([0, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 340 < X_pos <= 350:
        #         delta = np.array([0, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-70, 130, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-70, 125, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-70, 105, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-70, 125, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-70, 125, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-70, 125, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-70, 185, 0, 0])
        #     elif 350 < X_pos <= 360:
        #         delta = np.array([0, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([-30, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #     elif 360 < X_pos <= 370:
        #         delta = np.array([0, 0, 0, 0])
        #         if -180 < Y_pos <= -170:
        #             delta = np.array([0, 110, 0, 0])
        #         if -190 < Y_pos <= -180:
        #             delta = np.array([-30, 143, 0, 0])
        #         if -200 < Y_pos <= -190:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -210 < Y_pos <= -200:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -220 < Y_pos <= -210:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -230 < Y_pos <= -220:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -240 < Y_pos <= -230:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -250 < Y_pos <= -240:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -260 < Y_pos <= -250:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -270 < Y_pos <= -260:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -280 < Y_pos <= -270:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -290 < Y_pos <= -280:
        #             delta = np.array([-30, 150, 0, 0])
        #         if -300 < Y_pos <= -290:
        #             delta = np.array([-30, 185, 0, 0])
        #
        #     # elif -255 <= Y_pos <= -245:
        #     #     delta = np.array([-55, 70, 27, 0])
        #     #     if -275 <= Y_pos <= -265:
        #     #         delta = np.array([-55, 70, 27, 0])
        #     #     print("1")
        #     # elif -225 <= Y_pos <= -215:
        #     #     delta = np.array([-17, 113, 83, 0])  # mm
        #     #     if 300 <= X_pos <= 310:
        #     #         delta = np.array([-35, 135, 83, 0])  # mm
        #     #     if 280 <= X_pos <= 290:
        #     #         delta = np.array([-165, 130, 83, 0])  # mm
        #     #         print("2")
        #     #     if -270 <= Y_pos <= -260:
        #     #         delta = np.array([-17, 110, 83, 0])  # mm
        #     #         print("2+")
        #     #     print("2")
        #     # elif 310 <= X_pos <= 340:
        #     #     delta = np.array([-50, 112, 50, 0])  # mm
        #     #     if -202 <= Y_pos <= -192:
        #     #         delta = np.array([-50, 135, 0, 0])
        #     #     if 310 <= X_pos <= 320: #??
        #     #         delta = np.array([-30, 112, 50, 0])  # mm
        #     #     if -265 <= Y_pos <= -256:
        #     #         delta = np.array([-17, 82, 50, 0])  # mm [-43, 72, 50, 0]
        #     #         print("9")
        #     #     if -255 <= Y_pos <= -245:
        #     #         delta = np.array([-17, 150, 50, 0])  # mm
        #     #         print("9+")
        #     #     print("9")
        #     # elif -105 <= Z_pos <= -102 and -242 <= Y_pos <= -232 and 299 <= X_pos <= 310:
        #     #     delta = np.array([-14, 118, 63, 0])  # mm
        #     #     print("8")
        #     # elif 310 <= X_pos <= 320: #??
        #     #     delta = np.array([-50, 80, 58, 0])
        #     #     if 300 <= X_pos <= 310:
        #     #         delta = np.array([-80, 100, 48, 0])
        #     #         print("3")
        #     #     if 285 <= X_pos < 300:
        #     #         delta = np.array([-80, 70, 48, 0])
        #     #         print("3+")
        #     #     if 254 <= X_pos < 264:
        #     #         delta = np.array([-32, 100, 48, 0])
        #     #     print("3")
        #     # elif -75 <= Y_pos <= -65 and 261 <= X_pos <= 271:
        #     #     delta = np.array([-60, -74, -222, 0])
        #     #     print("6")
        #     # elif -230 <= Y_pos <= -210:
        #     #     delta = np.array([-60, 160, -222, 0])
        #     #     if 230 <= X_pos <= 240:
        #     #         delta = np.array([-10, 160, -222, 0])
        #     #         if 260 <= X_pos <= 270:
        #     #             delta = np.array([-30, 160, -222, 0])
        #     #             if 290 <= X_pos <= 300:
        #     #                 delta = np.array([-70, 160, -222, 0])
        #     #     if 275 <= X_pos <= 290:
        #     #         delta = np.array([-20, 160, -222, 0])
        #     #         if -220 <= Y_pos <= -210:
        #     #             delta = np.array([-20, 130, -222, 0])
        #     #         if -231 <= Y_pos <= -220:
        #     #             delta = np.array([-20, 120, -222, 0])
        #     #     if 255 <= X_pos <= 270:
        #     #         delta = np.array([-5, 160, -222, 0])
        #     #         if -231 <= Y_pos <= -220:
        #     #             delta = np.array([-5, 122, -222, 0])
        #     #     print("5")
        #     # elif 280 <= X_pos <= 290:
        #     #     delta = np.array([-165, 100, 83, 0])
        #     #     if -260 <= Y_pos <= -250:
        #     #         delta = np.array([-165, 130, 83, 0])
        #     # #     if -620 <= Y_pos <= -600:
        #     # #         delta = np.array([-67, 502, 83, 0])
        #     # #         if 280 <= X_pos <= 290:
        #     # #             delta = np.array([-8, 502, 83, 0])
        #     # #     if -620 <= Y_pos <= -610:
        #     # #         delta = np.array([-67, 474, 83, 0])
        #     # #         if 280 <= X_pos <= 290:
        #     # #             delta = np.array([-8, 474, 83, 0])
        #     #         print("10+")
        #     #     print("10")
        #     # elif -290 <= Y_pos <= -270:
        #     #     delta = np.array([-35, 150, 58, 0]) # 213, -131, -43
        #     #     print("11")
        #     # elif -90 <= Y_pos <= -70:
        #     #     delta = np.array([-35, -50, 58, 0])
        #     #     if 260 <= X_pos <= 270:
        #     #         delta = np.array([-12, -50, 58, 0])
        #     #     print("12")
        #     # elif 255 <= X_pos <= 265:
        #     #     delta = np.array([-5, 125, 58, 0])
        #     #     print("14")
        #     # elif 300 <= X_pos <= 310:
        #     #     delta = np.array([-61, 146, 58, 0])  # mm
        #     #     if -239 <= Y_pos <= -229:
        #     #         delta = np.array([-35, 130, 58, 0])
        #     #     print("15")
        #     # elif -235 <= Y_pos <= -225:
        #     #     delta = np.array([-35, 105, 58, 0])
        #     #     if 281 <= X_pos <= 291:
        #     #         delta = np.array([-5, 135, 58, 0])
        #     #     print("4")
        #     # elif 292 <= X_pos <= 302:
        #     #     delta = np.array([-20, 98, 58, 0])
        #     #     print("16")
        #     else:
        #         delta = np.array([-40, 130, 0, 0])  # mm
        #         print("7")
        #     # vi tri XYZ
        #     self.Matrix_pos = self.Matrix_pos + delta.reshape(-1, 1)
        #     while not 195 <= float(self.Matrix_pos[0][0]) <= 295:
        #         if float(self.Matrix_pos[0][0]) > 295:
        #             self.Matrix_pos[0] -= 10
        #         if float(self.Matrix_pos[0][0]) < 195:
        #             self.Matrix_pos[0] += 15
        #     while not -180 <= float(self.Matrix_pos[1][0]) <= -50:
        #         if float(self.Matrix_pos[1][0]) >= -50:
        #             self.Matrix_pos[1] -= 20
        #         if float(self.Matrix_pos[1][0]) <= -180:
        #             self.Matrix_pos[1] += 20
        #     while not -45 <= float(self.Matrix_pos[2][0]) <= -40:
        #         if float(self.Matrix_pos[2][0]) >= -40:
        #             self.Matrix_pos[2] -= 3
        #         if float(self.Matrix_pos[2][0]) <= -45:
        #             self.Matrix_pos[2] += 3
        #     print("Ma tran vi tri:", self.Matrix_pos)
        #
        #     # goc roll, pitch, yaw
        #     if X_pos == X_pos and xpos == 1:
        #         delta_angle = np.array([0, 0, 0])
        #     elif Pitch_pos > 19:
        #         delta_angle = np.array([0, 0, 0])
        #         if Roll_pos <= -150 or 150 <= Roll_pos:
        #             delta_angle = np.array([0, 0, 0])
        #             print("6")
        #         if 70 <= Pitch_pos <= 80:
        #             delta_angle = np.array([0, -60, 0])
        #         self.Matrix_angle[1] = - float(self.Matrix_angle[1][0])
        #         print("3+")
        #     elif -80 <= Pitch_pos <= -70 and Yaw_pos < 40:
        #         delta_angle = np.angle([0, 66, 0])
        #         print("1+")
        #     elif -180 < Roll_pos < -90:
        #         delta_roll = -90 - Roll_pos
        #         self.Matrix_angle[0] = 180 - delta_roll
        #         delta_angle = np.array([0, 0, 0])
        #         print("4")
        #         if -30 <= Pitch_pos <= -20:
        #             delta_angle = np.array([0, 10, 0])
        #         if -40 <= Pitch_pos <= -30:
        #             delta_angle = np.array([0, 20, 0])
        #     elif Roll_pos <= -140 or 140 <= Roll_pos:
        #         delta_angle = np.array([0, 0, 0])
        #         print("5")
        #         if -30 <= Pitch_pos <= -20:
        #             delta_angle = np.array([0, 10, 0])
        #         if -40 <= Pitch_pos <= -30:
        #             delta_angle = np.array([0, 20, 0])
        #     else:
        #         delta_angle = np.array([-93, 0, 0]) # [0, 0, 90]
        #         print("2+")
        #     self.Matrix_angle = self.Matrix_angle + delta_angle.reshape(-1, 1)
        #     # roll > 140 and roll < -140
        #     while -140 <= float(self.Matrix_angle[0][0]) <= 140:
        #         if 0.001 <= float(self.Matrix_angle[0][0]) < 140:
        #             self.Matrix_angle[0] += 5
        #         if -140 < float(self.Matrix_angle[0][0]) <= -0.001:
        #             self.Matrix_angle[0] -= 5
        #     while not -10 <= float(self.Matrix_angle[1][0]) <= 10: # pitch
        #         if float(self.Matrix_angle[1][0]) > 10:
        #             self.Matrix_angle[1][0] -= 2
        #         if float(self.Matrix_angle[1][0]) < -10:
        #             self.Matrix_angle[1][0] += 2
        #     self.Matrix_angle[2] = self.interpolation() # noi suy tuyen tinh Yaw
        #     print("Ma tran goc xoay:", self.Matrix_angle)

        #in vi tri doi tuong
        self.X_OBJ.setText(str(self.Matrix_pos[0]))
        self.Y_OBJ.setText(str(self.Matrix_pos[1]))
        self.Z_OBJ.setText(str(self.Matrix_pos[2]))
        self.Roll_OBJ.setText(str(self.Matrix_angle[0]))
        self.Pitch_OBJ.setText(str(self.Matrix_angle[1]))
        self.Yaw_OBJ.setText(str(self.Matrix_angle[2]))

    def start_pick_and_place(self):
        # if stage 1 chay toi vi tri vat -> stage rong -> stage 2 xuong Z -> stage rong
        # stage 3 gap -> rong -> stage 4 di toi vi tri tha vat -> rong -> tha vat -> rong -> ve home
        if self.stage == 0:
            QTimer.singleShot(0, self.Pick)
        elif self.stage == 1:
            QTimer.singleShot(2500, self.function_rong)
        elif self.stage == 2:
            QTimer.singleShot(0, self.Move_Z_pos_down)
        elif self.stage == 3:
            QTimer.singleShot(3000, self.function_rong_1)
        elif self.stage == 4:
            QTimer.singleShot(500, self.send_gripper)
        elif self.stage == 5:
            QTimer.singleShot(2500, self.function_rong_2)
        elif self.stage == 6:
            QTimer.singleShot(0, self.Move_Z_pos_up)
        elif self.stage == 7:
            QTimer.singleShot(2500, self.function_rong_3)
        elif self.stage == 8:
            if self.length < 60:
                QTimer.singleShot(0, self.Place_obj_duck)
            else:
                QTimer.singleShot(0, self.Place_obj_seal)
        elif self.stage == 9:
            QTimer.singleShot(3000, self.function_rong_4)
        elif self.stage == 10:
            QTimer.singleShot(0, self.send_gripper_release)
        elif self.stage == 11:
            QTimer.singleShot(2500, self.function_rong_5)
        elif self.stage == 12:
            QTimer.singleShot(0, self.Home)
        elif self.stage == 13:
            QTimer.singleShot(5000, self.function_rong_6)

    def Pick(self):
        self.object_flag = 1
        if self.animal_class == "duck":
            self.connection.moveCartasianPos(int(2000),
                                             int(self.Matrix_pos[0][0]) * 1000,
                                             int(self.Matrix_pos[1][0]) * 1000,
                                             int(self.Matrix_pos[2][0]) * 1000,
                                             int(self.Matrix_angle[0][0]) * 10000,
                                             int(self.Matrix_angle[1][0]) * 10000,
                                             int(self.Matrix_angle[2][0]) * 10000)
        elif self.animal_class == "seal":
            self.connection.moveCartasianPos(int(2000),
                                             int(self.Matrix_pos[0][0]) * 1000,
                                             int(self.Matrix_pos[1][0]) * 1000,
                                             int(self.Matrix_pos[2][0]) * 1000,
                                             int(self.Matrix_angle[0][0]) * 10000,
                                             int(self.Matrix_angle[1][0]) * 10000,
                                             int(self.Matrix_angle[2][0]) * 10000)
        # print("pick", self.stage)
        self.stage = 1
        self.start_pick_and_place()

    def function_rong(self):
        # print("rong", self.stage)
        self.stage = 2
        self.start_pick_and_place()

    def Move_Z_pos_down(self):
        # print("Executing Move_Z_pos_down", self.stage)
        self.speed = int(2000)
        self.z = -int(20 * 1000) if self.length < 60 else -int(35 * 1000)
        self.connection.moveCartasianPosIncre(self.speed, 0, 0, self.z, 0, 0, 0)
        self.stage = 3
        self.start_pick_and_place()

    def function_rong_1(self):
        # print("rong", self.stage)
        self.stage = 4
        self.start_pick_and_place()

    def send_gripper(self):
        # print("gap1", self.stage)
        if self.length < 60:
            self.send_grasp_angle_duck(-100)
        else:
            self.send_grasp_angle_seal(-100)
        self.stage = 5
        self.start_pick_and_place()

    def function_rong_2(self):
        # print("rong", self.stage)
        self.stage = 6
        self.start_pick_and_place()

    def Move_Z_pos_up(self):
        # print("Executing Move_Z_pos_up", self.stage)
        self.speed = int(2000)
        self.z = int(70 * 1000) if self.length < 60 else int(70 * 1000)
        self.connection.moveCartasianPosIncre(self.speed, 0, 0, self.z, 0, 0, 0)
        self.stage = 7
        self.start_pick_and_place()

    def function_rong_3(self):
        # print("rong", self.stage)
        self.stage = 8
        self.start_pick_and_place()

    def Place_obj_duck(self):
        # print("tha vat", self.stage)
        print("length", self.length)
        self.connection.moveCartasianPos(2000, int(-57.403) * 1000,
                                         int(172.719) * 1000, int(-7.777) * 1000,
                                         180 * 10000, 0, 0)
        self.stage = 9
        self.start_pick_and_place()

    def Place_obj_seal(self):
        # print("tha vat", self.stage)
        self.connection.moveCartasianPos(2000, int(55.598) * 1000,
                                         int(186.123) * 1000, int(-7.777) * 1000,
                                         180 * 10000, 0, 0)
        self.stage = 9
        self.start_pick_and_place()

    def function_rong_4(self):
        # print("rong", self.stage)
        self.stage = 10
        self.start_pick_and_place()

    def send_gripper_release(self):
        # print("tha", self.stage)
        if self.length < 60:
            self.send_grasp_angle_duck(110)
        else:
            self.send_grasp_angle_seal(110)
        self.stage = 11
        self.start_pick_and_place()

    def function_rong_5(self):
        # print("rong", self.stage)
        self.stage = 12
        self.start_pick_and_place()

    def Home(self):
        # print("ve home", self.stage)
        self.object_flag = 0
        self.connection.movePulsePos(2000, 0, 0, 0, 0, 0, 0)
        self.stage = 13
        self.start_pick_and_place()

    def function_rong_6(self):
        # print("rong", self.stage)
        self.stage = 0
        self.start_pick_and_place()

    def interpolation(self):
        # Dữ liệu cặp (A, B)
        A_values = [-92.052, 38.942, -16.54, -114.93, -361.99, 360.99]
        B_values = [-51.265, -44.3941, 26.5772, -77.1435, 0, 0]

        # Tạo hàm nội suy
        interpolate = interp1d(A_values, B_values, kind='linear')

        A = self.Matrix_angle[2]
        B = interpolate(A)
        return B

    def interpolation_duck_x(self):
        A = [302.14, 205.34, 236.091, 345.84, 293.61, 269.2, 251.83, 190]
        B = [282.61, 195.06, 206.09, 275.85, 286.27, 264.21, 221.84, 170]

        interpolate = interp1d(A, B, kind="linear")

        A = self.Matrix_pos[0]
        B = interpolate(A)

        return B

    def interpolation_seal_yaw(self):
        A = [-180, -90, -60, -143.99, -225, 40, 180]
        B = [-90, 0, -30, 42, -40, 35, -90]

        interpolate = interp1d(A, B, kind="linear")

        A = self.Matrix_angle[2]
        B = interpolate(A)

        return B


def UIbuild():
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = GUI(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    UIbuild()




           
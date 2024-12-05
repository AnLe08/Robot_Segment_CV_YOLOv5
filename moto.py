from asyncio.windows_events import NULL
import socket
from tkinter import FIRST
from PyQt5.QtCore import QByteArray

from numpy import int32, uint, uint16, uint32, uint8, size
#from testrun_gripper import *
from testrun import *
from structure import *

class Motomini:
    def __init__(self) -> None:
        self.sever_socket = socket.socket() # tạo giao thức truyền thông
        # self.sever_socket = QUdpSocket()
        self.connectState: bool = False
        self.servoState: bool = False
        self.ip: str = ""
        # self.ip = QHostAddress()
        self.port: int = 0

        self.rx_buffer: QByteArray = QByteArray()
        self.rx_buffer_pulse: QByteArray = QByteArray()
        self.rx_buffer_cartesian: QByteArray = QByteArray()

    def connectMotomini(self, ip: str, port: int):
        self.sever_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.ip = ip
        self.port = port
        self.sever_socket.connect((self.ip, self.port))
        # self.sever_socket.bind(self.ip,self.port)
        self.connectState = True

    def disconnectMotomini(self):
        self.sever_socket.close()
        self.ip = ""
        self.port = 0
        self.connectState = False

    def checkConnectState(self):
        return self.connectState

    def checkServoState(self):
        return self.servoState

    def sendData(self, buffer: QByteArray):
        self.sever_socket.sendto(buffer, (self.ip, self.port))
        # self.sever_socket.writeDatagram(buffer,self.ip,self.port)

    def receiveData(self):
        self.sever_socket.settimeout(500)
        try:
            self.rx_buffer = self.sever_socket.recv(520) # nhận dữ liệu
            if self.rx_buffer[11] == 4:
                self.rx_buffer_cartesian = self.rx_buffer
            elif self.rx_buffer[11] == 6:
                self.rx_buffer_pulse = self.rx_buffer
        except socket.timeout:
            pass
        # while self.sever_socket.hasPendingDatagrams():
        #     self.rx_buffer,size,addr = self.sever_socket.readDatagram(520)
        #     print(self.rx_buffer)
        #     if self.rx_buffer[11] == 2:
        #         self.rx_buffer_cartesian = self.rx_buffer
        #     elif self.rx_buffer[11] == 3:
        #         self.rx_buffer_pulse = self.rx_buffer
    
    def onServo(self) -> int:
        header = txHeader
        header.id = receiveType.ON_SERVO.value
        header.command_no = 0x83
        header.instance = 2         # this is not the value this stand for the byte of itself
        header.attribute = 1        # this is not the value this stand for the byte of itself
        header.service = 0x10
        header.data_size = 4        # this is not the value this stand for the byte of itself

        buffer = QByteArray()
        buffer = header.returnByteArray()
        buffer.append(QByteArray(struct.pack("<I", 1))) 
        self.sendData(buffer=buffer)
        self.receiveData()
        if (self.rx_buffer[11] == 0) & (int.from_bytes(self.rx_buffer[26:28],"big") == 0): #rx_buffer[11] = header.id, rx_buffer[26:28] = header.padding
            self.servoState = True
            return 0
        else:
            return 1

    def offServo(self):
        header = txHeader
        header.id = receiveType.OFF_SERVO.value
        header.command_no = 0x83
        header.instance = 2
        header.attribute = 1
        header.service = 0x10
        header.data_size = 4

        buffer = QByteArray()
        buffer = header.returnByteArray() # this is the structure frame of the header part will be send
        buffer.append(QByteArray(struct.pack("<I", 2)))# this will be attach to the structure frame we can also call it the datapart
        self.sendData(buffer=buffer)
        self.receiveData()
        if (self.rx_buffer[11] == 1) & (int.from_bytes(self.rx_buffer[26:28],"big") == 0):
            self.servoState = False
            return 0
        else:
            return 1

    #class homeServo(self):

    def moveCartasianPos(self, speed: uint32,X: int32,Y: int32,Z: int32,Roll: int32,Pitch: int32,Yaw: int32):
        #X: int32,Y: int32,Z: int32,Roll: int32,Pitch: int32,Yaw: int32
        data = txPosition
        data.classification_in_speed = 0
        data.X = X
        #data.X = pos[0]
        data.Y = Y
        #data.X = pos[1]
        data.Z = Z
        #data.X = pos[2]
        data.Roll = Roll
        #data.Roll = pos[3]
        data.Pitch = Pitch
        #data.Pitch = pos[4]
        data.Yaw = Yaw
        #data.Yaw = pos[5]
        data.speed = speed

        header = txHeader
        header.id = receiveType.WRITE_POSITION.value
        header.command_no = 0x8A
        header.instance = 0x01
        header.attribute = 0x01
        header.service = 0x02
        header.data_size = data.size

        buffer = QByteArray()
        buffer = header.returnByteArray()
        buffer.append(data.returnByteArray())
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

    def moveCartasianStraight(self, speed: uint32,X: int32,Y: int32,Z: int32,Roll: int32,Pitch: int32,Yaw: int32):
        #X: int32,Y: int32,Z: int32,Roll: int32,Pitch: int32,Yaw: int32
        data = txPosition
        data.classification_in_speed = 0x01
        data.X = X
        #data.X = pos[0]
        data.Y = Y
        #data.X = pos[1]
        data.Z = Z
        #data.X = pos[2]
        data.Roll = Roll
        #data.Roll = pos[3]
        data.Pitch = Pitch
        #data.Pitch = pos[4]
        data.Yaw = Yaw
        #data.Yaw = pos[5]
        data.speed = speed

        header = txHeader
        header.id = receiveType.WRITE_POSITION.value
        header.command_no = 0x8A
        header.instance = 0x02
        header.attribute = 0x01
        header.service = 0x02
        header.data_size = data.size

        buffer = QByteArray()
        buffer = header.returnByteArray()
        buffer.append(data.returnByteArray())
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

    def moveCartasianPosIncre(self, speed: uint32,X: int32,Y: int32,Z: int32,Roll: int32,Pitch: int32,Yaw: int32):
        #X: int32,Y: int32,Z: int32,Roll: int32,Pitch: int32,Yaw: int32
        data = txPosition
        data.classification_in_speed = 0x01
        data.X = X
        #data.X = pos[0]
        data.Y = Y
        #data.X = pos[1]
        data.Z = Z
        #data.X = pos[2]
        data.Roll = Roll
        #data.Roll = pos[3]
        data.Pitch = Pitch
        #data.Pitch = pos[4]
        data.Yaw = Yaw
        #data.Yaw = pos[5]
        data.speed = speed

        header = txHeader
        header.id = receiveType.WRITE_INCPOSITION.value
        header.command_no = 0x8A
        header.instance = 0x03
        header.attribute = 0x01
        header.service = 0x02
        header.data_size = data.size

        buffer = QByteArray()
        buffer = header.returnByteArray()
        buffer.append(data.returnByteArray())
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0
    
    def movePulsePos(self, speed: uint32,S : int32,L: int32,U: int32,R: int32,B: int32,T: int32):
        data = txPulse
        data.classification_in_speed = 0
        data.robot_1_pulse = S
        data.robot_2_pulse = L
        data.robot_3_pulse = U
        data.robot_4_pulse = R
        data.robot_5_pulse = B
        data.robot_6_pulse = T
        data.speed = speed

        header = txHeader
        header.id = receiveType.WRITE_PUSLE.value
        header.command_no = 0x8B
        header.instance = 0x01
        header.attribute = 0x01
        header.service = 0x02
        header.data_size = data.size

        buffer = QByteArray()
        buffer = header.returnByteArray()
        buffer.append(data.returnByteArray())
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

    #def GoHome(self):
        #data = txPulse
        #data.r1 = 0
        #data.r2 = 0
        #data.r3 = 0
        #data.r4 = 0
        #data.r5 = 0
        #data.r6 = 0
        #data.speed = 10*100

        #header = txHeader
        #header.id = receiveType.HOME_SERVO.value
        #header.command_no = 0x8B
        #header.instance = 0x01
        #header.attribute = 0x01
        #header.service = 0x02
        #header.data_size = data.size

        #buffer = QByteArray()
        #buffer = header.returnByteArray()
        #buffer.append(data.returnByteArray())
        #self.sendData(buffer=buffer)
        #self.receiveData()
        #return 0

    def getVariablePos(self):
        #, first: uint32, second: uint32, third: uint32, fourth: uint32, fifth: uint32, sixth: uint32
        data = txVariablePosition
        data.data_type= 0x10
        data.Type: uint32 = 0
        data.tool_number: uint32 = 0
        data.user_coordinate_number: uint32 = 0
        data.extended_type: uint32 = 0
        data.axis1_data = 0
        data.axis2_data = 0
        data.axis3_data = 0
        data.axis4_data = 0
        data.axis5_data = 0
        data.axis6_data = 0
        data.axis7_data = 0
        data.axis8_data = 0

        header = txHeader
        header.id = receiveType.GET_POSITION.value
        header.command_no = 0x75
        header.instance = 0x65
        header.attribute = 0
        header.service = 0x01
        header.data_size = 0

        buffer = QByteArray()
        buffer = header.returnByteArray()
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

    def getVariablePulse(self):
        #, first: uint32, second: uint32, third: uint32, fourth: uint32, fifth: uint32, sixth: uint32
        data = txVariablePosition
        data.data_type= 0x00
        data.Type: uint32 = 0
        data.tool_number: uint32 = 0
        data.user_coordinate_number: uint32 = 0
        data.extended_type: uint32 = 0
        data.axis1_data = 0
        data.axis2_data = 0
        data.axis3_data = 0
        data.axis4_data = 0
        data.axis5_data = 0
        data.axis6_data = 0
        data.axis7_data = 0
        data.axis8_data = 0

        header = txHeader
        header.id = receiveType.GET_PULSE.value
        header.command_no = 0x75
        header.instance = 0x01
        header.attribute = 0
        header.service = 0x01
        header.data_size = 0

        buffer = QByteArray()
        buffer = header.returnByteArray()
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

    def getByte(self, index: uint16):
        header = txHeader
        header.id = receiveType.GET_BYTE.value
        header.command_no = 0x7A
        header.instance = index
        header.attribute = 0x01
        header.service = 0x0E
        header.data_size = 0

        buffer = QByteArray
        buffer = header.returnByteArray()
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

    def writeByte(self, index: uint16, var: uint8):
        header = txHeader
        header.id = receiveType.WRITE_BYTE.value
        header.command_no = 0x7A
        header.instance = index
        header.attribute = 0x01
        header.service = 0x10
        header.data_size = 1

        buffer = QByteArray()
        buffer = header.returnByteArray()
        buffer.append(QByteArray(struct.pack("B", var)))
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

    def state_Robot(self):
        header = txHeader
        header.id = receiveType.READ_STATUS.value
        header.command_no = 0x72
        header.instance = 1
        header.attribute = 1
        header.service = 0x01
        header.data_size = 0

        buffer = QByteArray()
        buffer = header.returnByteArray()
        self.sendData(buffer=buffer)
        self.receiveData()
        return 0

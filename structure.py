from enum import Enum
import enum
from PyQt5.QtCore import QByteArray
import struct
from numpy import int32, uint, uint16, uint32, uint8, size

class cordinateVariable:
    S_pulse_degree = 34816/30
    L_pulse_degree = 102400/90
    U_pulse_degree = 102400/90
    RBT_pulse_degree = 102400/90
    CartesianPos: list = [0,0,0,0,0,0]
    PulsePos: list = [0,0,0,0,0,0]



class receiveType(Enum):
    ON_SERVO = 0x00
    OFF_SERVO = 0x01
    HOME_SERVO = 0X02
    WRITE_POSITION = 0x03
    GET_POSITION = 0X04
    WRITE_PUSLE = 0x05
    GET_PULSE = 0X06
    WRITE_INCPOSITION = 0x07
    GET_BYTE = 0x0E
    WRITE_BYTE = 0x0F
    READ_STATUS = 0X10

class txHeader:
    header_size: uint16 = 0x20
    data_size: uint16 = 0
    reserve1:uint8 = 0x01 # fix to character "3" in ASCII mode
    processing_division: uint8 = 1
    ack: uint8 = 0
    id: uint8 = 0
    command_no: uint16 = 0
    instance: uint16 = 0
    attribute: uint8 = 0
    service: uint8 = 0
    padding: uint16 = 0

    def returnByteArray():
        buffer = QByteArray()
        buffer.append(
            QByteArray(
                struct.pack(
                    "<4s2H4BI8s2H2BH",# this define the datatype below <4s/2H/4B/I/8s/2H/2B/H
                    b"YERC",                        #4 byte, data type : 4s / fix to character "YERC" in ASCII mode
                    txHeader.header_size,           #2 byte, data type : H
                    txHeader.data_size,             #2 byte, data type : H
                    txHeader.reserve1,              #1 byte, data type : B / fix to character "3" in ASCII mode
                    txHeader.processing_division,   #1 byte, data type : B
                    txHeader.ack,                   #1 byte, data type : B
                    txHeader.id,                    #1 byte, data type : B
                    0,                              #4 byte, data type : I
                    b"99999999",                    #8 byte, data type : 8s / fix to character "99999999" in ASCII mode
                    txHeader.command_no,            #2 byte, data type : H
                    txHeader.instance,              #2 byte, data type : H
                    txHeader.attribute,             #1 byte, data type : B
                    txHeader.service,               #1 byte, data type : B
                    txHeader.padding,               #2 byte, data type : H
                )
            )
        )
        return buffer

class txVariablePosition():
    data_type: uint32 = 0
    Type: uint32 = 0
    tool_number: uint32 = 0
    user_coordinate_number: uint32 = 0
    extended_type: uint32 = 0
    axis1_data: uint32 = 0
    axis2_data: uint32 = 0
    axis3_data: uint32 = 0
    axis4_data: uint32 = 0
    axis5_data: uint32 = 0
    axis6_data: uint32 = 0
    axis7_data: uint32 = 0
    axis8_data: uint32 = 0
    size = 52 #4byte*13

    def returnByteArray():
        buffer = QByteArray()
        buffer.append(
            QByteArray(
                struct.pack(
                    "<5I6i15I",
                    txVariablePosition.data_type,               #datatype : I
                    txVariablePosition.Type,                    #datatype : I
                    txVariablePosition.tool_number,             #datatype : I
                    txVariablePosition.user_coordinate_number,  #datatype : I
                    txVariablePosition.extended_type,           #datatype : I
                    txVariablePosition.axis1_data,              #datatype : i
                    txVariablePosition.axis2_data,              #datatype : i
                    txVariablePosition.axis3_data,              #datatype : i
                    txVariablePosition.axis4_data,              #datatype : i
                    txVariablePosition.axis5_data,              #datatype : i            
                    txVariablePosition.axis6_data,              #datatype : i                
                    txVariablePosition.axis7_data,              #datatype : i
                    txVariablePosition.axis8_data2,             #datatype : i
                )
            )
        )
        return buffer

class txPosition():
    control_group_robot: uint32 = 1
    control_group_station: uint32 = 0
    classification_in_speed: uint32 = 0
    speed: uint32 = 0
    cordinate_operation: uint32 = 0x10
    X: int32 = 0
    Y: int32 = 0
    Z: int32 = 0
    Roll: int32 = 0
    Pitch: int32 = 0
    Yaw: int32 = 0
    reservation1: uint32 = 0
    reservation2: uint32 =0
    Type: uint32 = 0
    expanded_Type: uint32 = 0
    tool_no: uint32 = 0
    user_cordinate_no: uint32 = 0
    b1_position: uint32 = 0
    b2_position: uint32 = 0
    b3_position: uint32 = 0
    s1_position: uint32 = 0
    s2_position: uint32 = 0
    s3_position: uint32 = 0
    s4_position: uint32 = 0
    s5_position: uint32 = 0
    s6_position: uint32 = 0
    size = 104 #4byte*26

    def returnByteArray():
        buffer = QByteArray()
        buffer.append(
            QByteArray(
                struct.pack(
                    "<5I6i15I",
                    txPosition.control_group_robot,             #datatype : I
                    txPosition.control_group_station,           #datatype : I
                    txPosition.classification_in_speed,         #datatype : I
                    txPosition.speed,                           #datatype : I
                    txPosition.cordinate_operation,             #datatype : I
                    txPosition.X,                               #datatype : i
                    txPosition.Y,                               #datatype : i
                    txPosition.Z,                               #datatype : i
                    txPosition.Roll,                            #datatype : i
                    txPosition.Pitch,                           #datatype : i
                    txPosition.Yaw,                             #datatype : i
                    txPosition. reservation1,                   #datatype : I
                    txPosition. reservation2,                   #datatype : I
                    txPosition.Type,                            #datatype : I
                    txPosition.expanded_Type,                   #datatype : I
                    txPosition.tool_no,                         #datatype : I
                    txPosition.user_cordinate_no,               #datatype : I
                    txPosition.b1_position,                     #datatype : I
                    txPosition.b2_position,                     #datatype : I
                    txPosition.b3_position,                     #datatype : I
                    txPosition.s1_position,                     #datatype : I
                    txPosition.s2_position,                     #datatype : I
                    txPosition.s3_position,                     #datatype : I
                    txPosition.s4_position,                     #datatype : I
                    txPosition.s5_position,                     #datatype : I
                    txPosition.s6_position,                     #datatype : I
                )
            )
        )
        return buffer

class txPulse():
    control_group_robot: uint32 = 1
    control_group_station: uint32 = 0
    classification_in_speed: uint32 = 0
    speed: uint32 = 0
    robot_1_pulse: int32 = 0
    robot_2_pulse: int32 = 0
    robot_3_pulse: int32 = 0
    robot_4_pulse: int32 = 0
    robot_5_pulse: int32 = 0
    robot_6_pulse: int32 = 0
    robot_7_pulse: int32 = 0
    robot_8_pulse: int32 = 0
    tool_no: uint32 = 0
    b1_position: uint32 = 0
    b2_position: uint32 = 0
    b3_position: uint32 = 0
    s1_position: uint32 = 0
    s2_position: uint32 = 0
    s3_position: uint32 = 0
    s4_position: uint32 = 0
    s5_position: uint32 = 0
    s6_position: uint32 = 0
    size = 88#4bytes*22

    def returnByteArray():
        buffer = QByteArray()
        buffer.append(
            QByteArray(
                struct.pack(
                    "<4I8i10I",
                    txPulse.control_group_robot,             #datatype : I
                    txPulse.control_group_station,           #datatype : I
                    txPulse.classification_in_speed,         #datatype : I
                    txPulse.speed,                           #datatype : I
                    txPulse.robot_1_pulse,                   #datatype : i
                    txPulse.robot_2_pulse,                   #datatype : i
                    txPulse.robot_3_pulse,                   #datatype : i
                    txPulse.robot_4_pulse,                   #datatype : i
                    txPulse.robot_5_pulse,                   #datatype : i
                    txPulse.robot_6_pulse,                   #datatype : i
                    txPulse.robot_7_pulse,                   #datatype : i
                    txPulse.robot_8_pulse,                   #datatype : i
                    txPulse.tool_no,                         #datatype : I
                    txPulse.b1_position,                     #datatype : I
                    txPulse.b2_position,                     #datatype : I
                    txPulse.b3_position,                     #datatype : I
                    txPulse.s1_position,                     #datatype : I
                    txPulse.s2_position,                     #datatype : I
                    txPulse.s3_position,                     #datatype : I    
                    txPulse.s4_position,                     #datatype : I        
                    txPulse.s5_position,                     #datatype : I    
                    txPulse.s6_position,                     #datatype : I
                )
            )
        )
        return buffer

class convertNameJob:
    job_name:str = ""
    def returnByteArray():
        list_name= []
        list_name[:0] = convertNameJob.job_name
        buffer = QByteArray()
        for i in range(size(list_name)):
            buffer.append(QByteArray(struct.pack("c",bytes(list_name[i], 'UTF-8'))))
        for i in range(size(list_name),32):
            buffer.append(QByteArray(struct.pack("B",0)))
        return buffer

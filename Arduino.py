import sys
import serial
import time

class ArduinoController:
    def __init__(self) -> None:
        self.serial_port: str = ""
        self.baud_rate: int = 0
        self.connection = None
        self.arduinoState: bool = False
        self.pre_command_seal = None
        self.pre_command_duck = None

    def connect_to_arduino(self, serial_port: str, baud_rate: int):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.connection = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        self.arduinoState = True
        print("Connected to Arduino")

    def disconnect_to_arduino(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            self.arduinoState = False
            print("Disconnected from Arduino")

    def checkConnectState(self):
        return self.arduinoState

    def send_command_seal(self, command):
        if self.connection and self.connection.is_open:
            # neu bam lan 1 gripper se chay "command" do
            # neu bam lan 2 gripper se quay lai vi tri cu
            command = int(command)
            # if command == self.pre_command_seal:
            #     command = -command

            self.connection.write(str(command).encode())
            # print(f"Sent command: {command}")

            self.pre_command_seal = command
        # else:
        #     print("Connection not open. Connect to Arduino first.")

    def send_command_duck(self, command):
        if self.connection and self.connection.is_open:
            # neu bam lan 1 gripper se chay "command" do
            # neu bam lan 2 gripper se quay lai vi tri cu
            command = int(command)
            # if command == self.pre_command_duck:
            #     command = -command

            self.connection.write(str(command).encode())
            # print(f"Sent command: {command}")

            self.pre_command_duck = command
        # else:
        #     print("Connection not open. Connect to Arduino first.")



import socket
import struct
import time
import numpy as np

class RealWorldDofbot():
    def __init__(self):
        IP = "192.168.72.165"
        PORT = 65432
        self.last_sync_time = 0
        self.sync_hz = 10000
        self.fail_quietly = False
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_address = (IP, PORT)
            print(server_address)
            self.sock.connect(server_address)
            print("Connected to real-world Dofbot!")
        except socket.error as e:
            self.failed = True
            print("Connection to real-world Dofbot failed!")
            if self.fail_quietely:
                print(e)
            else:
                raise e
    
    def send_joint_pos(self, servoAngle):
        if time.time() - self.last_sync_time < 1 / self.sync_hz:
            return
        self.last_sync_time = time.time()
        packer = struct.Struct('f f f f f f')
        servo_angles = servoAngle
        packed_data = packer.pack(*servo_angles)
        try:
            self.sock.sendall(packed_data)
        except socket.error as e:
            self.failed = True
            print("Send to real-world Dofbot failed!")
            if self.fail_quietely:
                print(e)
            else:
                raise e
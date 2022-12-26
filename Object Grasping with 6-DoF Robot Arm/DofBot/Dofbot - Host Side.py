import time
import binascii
import socket
import struct
from Arm_Lib import Arm_Device
import select
import threading
import time

# Fix Address already in use
import os
# Kill all process that are using TCP 65432 port
os.system('fuser -kn tcp 65432 | awk')

HOST = '' # Standard loopback interface address (localhost)
PORT = 65432 # Port to listen on (non-privileged ports are > 1023)

# Get DOFBOT object
Arm = Arm_Device()
time.sleep(0.1)

N_JOINTS = 6
state = [90] * N_JOINTS
DELAY = 0.01

def receiveTextViaSocket(sock):
    unpacker = struct.Struct('f f f f f f')

    connection, client_address = sock.accept()
    print('Connected:', client_address)
    while True:
        data = connection.recv(unpacker.size)
        if not data:
            # Connection closed by client
            connection.close()
            return
        stringifiedData = str(binascii.hexlify(data))

        if stringifiedData != "b''":
            unpacked_data = unpacker.unpack(data)
            assert len(unpacked_data) == N_JOINTS
            for i in range(N_JOINTS):
                state[i] = unpacked_data[i]

def start_server():
    print('server starting')
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (HOST, PORT)
    sock.bind(server_address)
    sock.listen(1)

    socks = [sock]
    while True:
        readySocks, _, _ = select.select(socks, [], [], 5)
        for sock in readySocks:
            receiveTextViaSocket(sock)
            print('connection closed')
        time.sleep(DELAY)

try:
    t = threading.Thread(target=start_server)
    t.start()
    while True:
        tme = 300
        msg = state + [tme]
        print(msg)
        time.sleep(DELAY)
except KeyboardInterrupt:
    print('KeyboardInterrupt')

del Arm  # Release DOFBOT object
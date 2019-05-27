from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('/home/cobot/code/caid2.0/python/Algorithm/maskrcnn-benchmark-stable/')

import contextlib
import json
import mmap
import socket
import threading
import re
import logging
from ServerDL.cfgs.Protocol import *
from ServerDL.apis.ExceptionHandler import ExceptionSendBack, HandleException
from ServerDL.cfgs.configfiles import *
from ServerDL.apis.DetectronFactory import DetectronFactory



digit = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


class DLServer:
    '''深度学习服务器端入口程序
    基于socket通信，基于共享内存传递图片，每一个客户端对象对应于服务端一条链接
    服务端每一条链接存在于一个单独的线程，并包含一个服务端对象'''

    def run(self):
        socketserver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socketserver.bind(('127.0.0.1', 9999))
        print(get_host_ip())
        socketserver.listen(1024)
        print('Waiting for connection...')
        while True:
            sock, addr = socketserver.accept()
            t = threading.Thread(target=self.tcplink, args=(sock, addr))
            t.start()

    def parse_pram(self, data):
        for key in data:
            if str.isnumeric(data[key]):
                data[key] = int(data[key])
            elif digit.match(data[key]):
                data[key] = float(data[key])
        return data

    def wrap_pram(self, data):
        for key in data:
            if isinstance(data[key], int):
                data[key] = str(data[key])
        return data

    def readbuffer(self, m_addr, rows, cols, channels):
        if not m_addr:
            raise NameError("invalid memory address")
        with open(m_addr, 'r+') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), rows * cols * channels)) as m:
                # 从共享内存中读出图片
                s = m.read(rows * cols * channels)

                a = np.fromstring(s, dtype=np.uint8)

                img = a.reshape((rows, cols, channels))
                return img

    def writebuffer(self, m_addr, image):
        if not m_addr:
            raise NameError("invalid memory address")
        rows, cols = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        with open(m_addr, 'r+') as f:
            with contextlib.closing(mmap.mmap(f.fileno(), rows * cols * channels)) as m:
                image_string = image.tostring()
                m.seek(0)
                m.write(image_string)

    def tcplink(self, sock, addr):
        '''每一条socket链接对应于一个客户端对象
        协议中共包含3种命令
        1. MODEL_INIT。该命令会导致服务端初始化一个新对象，该对象中包含一个或多个深度学习模型
                       该命令在客户端初始化时发送一次
        2. READ_BUFFER. 该命令用于客户端向服务端发送待预测的图片，该命令可连续调用，待预测图片会保存到列表中
        3. MODEL_PREDICT. 该命令用于客户端向服务端发送预测指令。 该命令每次预测一张图片，
                          客户端保证该命令会被连续调用，直至预测完列表中的所有图片'''
        imlist = []
        m_addr = ''
        detector = None
        print('Accept new connection from %s:%s...' % addr)
        # 进入循环，处理远程调用
        while True:

            data = sock.recv(65535).decode('utf-8')
            if not data or data == 'exit':
                break
            print('reply the client %s:%s.' % addr)
            new_dict = json.loads(data)
            new_dict = self.parse_pram(new_dict)
            print(new_dict)
            try:
                # VALUE.REQUEST_TYPE.MODEL_INIT 加载并初始化模型
                if VALUE.REQUEST_TYPE.MODEL_INIT == int(new_dict[KEY.REQUEST_TYPE]):
                    if detector is not None:
                        raise RuntimeError("The model detector is not None when model init.")
                    if imlist:
                        imlist.clear()
                    detector = DetectronFactory.get(new_dict)
                    m_addr = new_dict[KEY.PREDICTION.MEMORY_ADDR]

                    sock.send(json.dumps(self.wrap_pram(success_example_init)).encode('utf-8'))

                # VALUE.REQUEST_TYPE.READ_BUFFER 读取共享内存中的图片
                elif VALUE.REQUEST_TYPE.READ_BUFFER == int(new_dict[KEY.REQUEST_TYPE]):
                    image = self.readbuffer(m_addr,
                                            new_dict[KEY.PREDICTION.ROWS],
                                            new_dict[KEY.PREDICTION.COLS],
                                            new_dict[KEY.PREDICTION.CHANNELS])
                    imlist.append(image)
                    sock.send(json.dumps(self.wrap_pram(success_example_buffer)).encode('utf-8'))

                # VALUE.REQUEST_TYPE.MODEL_PREDICT 预测
                elif VALUE.REQUEST_TYPE.MODEL_PREDICT == int(new_dict[KEY.REQUEST_TYPE]):
                    if not detector:
                        raise RuntimeError("The model detector is None.")
                    if not imlist:
                        raise RuntimeError("The image List is empty.")

                    response_json, response_img = detector.predict(imlist.pop(), **new_dict)
                    if response_img is not None:
                        self.writebuffer(m_addr, response_img)
                    sock.send(json.dumps(self.wrap_pram(response_json)).encode('utf-8'))

                    print(len(imlist))

            except Exception as e:
                logging.exception(e)
                ExceptionSendBack(e, sock, new_dict[KEY.REQUEST_TYPE])
                HandleException(e, new_dict[KEY.REQUEST_TYPE])
                imlist.clear()

        sock.close()
        print('Connection from %s:%s closed.' % addr)


DLServer().run()
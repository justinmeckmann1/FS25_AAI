#!/usr/bin/env python3

import numpy as np
import cv2
import time
import onnxruntime as ort
import tensorrt as trt
import os
import torch
import pycuda.driver as cuda
import pycuda.autoinit

import argparse
import os
import io

import tornado.ioloop
import tornado.web
import tornado.websocket

import cv2
import base64

from processing import cifar_model_trt

parser = argparse.ArgumentParser(description='Start the PyImageStream server.')

parser.add_argument('--port', default=8080, type=int, help='Web server port (default: 8080)')
parser.add_argument('--camera', default=0, type=int, help='Camera index, first camera is 0 (default: 0)')
parser.add_argument('--width', default=640, type=int, help='Width (default: 640)')
parser.add_argument('--height', default=480, type=int, help='Height (default: 480)')
parser.add_argument('--quality', default=90, type=int, help='JPEG Quality 1 (worst) to 100 (best) (default: 70)')
parser.add_argument('--stopdelay', default=7, type=int, help='Delay in seconds before the camera will be stopped after '
                                                             'all clients have disconnected (default: 7)')
args = parser.parse_args()

class Camera:

    def __init__(self, index, width, height, quality, stopdelay):
        print("Initializing camera...")
        
        capture_device = 0

        cam = cv2.VideoCapture(capture_device)

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self._cam = cam
        print("Camera initialized")
        self.is_started = False
        self.stop_requested = False
        self.quality = quality
        self.stopdelay = stopdelay

        if False:
            self.fcn_model = fcn_model()
            self.fcn_model = fcn_model_trt()
        else:
            self.fcn_model = cifar_model_trt()

    def request_start(self):
        if self.stop_requested:
            print("Camera continues to be in use")
            self.stop_requested = False
        if not self.is_started:
            self._start()

    def request_stop(self):
        if self.is_started and not self.stop_requested:
            self.stop_requested = True
            print("Stopping camera in " + str(self.stopdelay) + " seconds...")
            tornado.ioloop.IOLoop.current().call_later(self.stopdelay, self._stop)

    def _start(self):
        print("Camera started")
        self.is_started = True

    def _stop(self):
        if self.stop_requested:
            print("Camera stopped")
            self.is_started = False
            self.stop_requested = False

    def get_jpeg_image_bytes(self):
        ret, frame = self._cam.read()
        proc_frame = self.fcn_model.predict(frame)
        result, im_arr = cv2.imencode('.jpg', proc_frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
    
        im_bytes = im_arr.tobytes()
        im_b64 = base64.b64encode(im_bytes)

        return im_bytes


camera = Camera(args.camera, args.width, args.height, args.quality, args.stopdelay)


class ImageWebSocket(tornado.websocket.WebSocketHandler):
    clients = set()

    def check_origin(self, origin):
        # Allow access from every origin
        return True

    def open(self):
        ImageWebSocket.clients.add(self)
        print("WebSocket opened from: " + self.request.remote_ip)
        camera.request_start()

    def on_message(self, message):
        jpeg_bytes = camera.get_jpeg_image_bytes()
        self.write_message(jpeg_bytes, binary=True)

    def on_close(self):
        ImageWebSocket.clients.remove(self)
        print("WebSocket closed from: " + self.request.remote_ip)
        if len(ImageWebSocket.clients) == 0:
            camera.request_stop()


script_path = os.path.dirname(os.path.realpath(__file__))
static_path = script_path + '/static/'

app = tornado.web.Application([
        (r"/websocket", ImageWebSocket),
        (r"/(.*)", tornado.web.StaticFileHandler, {'path': static_path, 'default_filename': 'index.html'}),
    ])
app.listen(args.port)

print("Starting server: http://localhost:" + str(args.port) + "/")

tornado.ioloop.IOLoop.current().start()

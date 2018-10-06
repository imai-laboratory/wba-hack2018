# -*- coding: utf-8 -*-
"""
 Main script for web monitor interface. 
"""

import os
import time
import base64
import argparse
import json
from threading import Lock
from http import HTTPStatus

import cv2

from multiprocessing import Process, Lock, Queue

import flask
import logging
from flask import Flask, make_response, send_from_directory
from jinja2 import FileSystemLoader
from werkzeug.local import Local, LocalManager
from flask_cors import CORS

#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)
app = Flask(__name__, static_url_path='')
CORS(app)
app.secret_key = 'oculomotor'
app.jinja_loader = FileSystemLoader(os.getcwd() + '/templates')


display_size = (128 * 4 + 16, 1500)

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model name", default=None)
parser.add_argument("--content", type=int, help="Model name", default=0)
args = parser.parse_args()


class Runner(object):
    def __init__(self):
        from inspector import Inspector
        from oculoenv import PointToTargetContent, ChangeDetectionContent, OddOneOutContent, VisualSearchContent, MultipleObjectTrackingContent, RandomDotMotionDiscriminationContent
        self.contents = [
            PointToTargetContent,
            ChangeDetectionContent,
            OddOneOutContent,
            VisualSearchContent,
            MultipleObjectTrackingContent,
            RandomDotMotionDiscriminationContent,
        ]
        self.content_id = args.content
        self.difficulty = -1
        self.inspector = Inspector(
            self.contents[args.content](-1), display_size,
            model_name=args.model_name
        )
        self.lock = Lock()

    def init(self):
        #self.set_content(0)
        #self.set_difficulty(-1)
        return self.info()

    def info(self):
        return json.dumps({
            'content_range': len(self.contents),
            'content': self.content_id,
            'difficulty_range': self.contents[self.content_id].difficulty_range,
            'difficulty': self.difficulty,
        })

    def step(self):
        #with self.lock:
        self.inspector.update()
        image = self.inspector.get_frame()
        data = cv2.imencode('.png', image)[1].tobytes()
        encoded = base64.encodestring(data)
        return encoded

    def set_content(self, content_id):
        #with self.lock:
        print('content')
        self.content_id = content_id
        content = self.contents[self.content_id](self.difficulty)
        self.inspector = Inspector(content, display_size)
        ret = {
            'difficulty_range': self.contents[content_id].difficulty_range,
            'difficulty': -1,
        }
        return flask.jsonify(ret)

    def set_difficulty(self, difficulty):
    #    with self.lock:
        print('difficulty')
        self.difficulty = difficulty
        content = self.contents[self.content_id](self.difficulty)
        self.inspector = Inspector(content, display_size)
        return 'New Content Created', HTTPStatus.OK


image_queue = Queue(10)
info_queue = Queue(10)
lock = Lock()
def training(image_queue, info_queue, lock):
    print('training start')
    runner = Runner()
    print('initialzie')
    runner.init()
    print('init')
    while True:
        image_string = runner.step()
        info = runner.info()
        lock.acquire()
        if image_queue.full():
            image_queue.get()
        image_queue.put(image_string)
        if info_queue.full():
            info_queue.get()
        info_queue.put(info)
        lock.release()

process = Process(target=training, args=(image_queue, info_queue, lock))
process.start()

@app.route('/init')
def init():
    info = info_queue.get()
    return info

@app.route('/info')
def info():
    info = info_queue.get()
    return info


@app.route('/step')
def step():
    image = image_queue.get()
    return make_response(image)


#@app.route('/content/<int:content_id>')
#def content(content_id):
#    return runner.set_content(content_id)
#
#@app.route('/difficulty/<difficulty>')
#def difficulty(difficulty):
#    return runner.set_difficulty(int(difficulty))


@app.route('/monitor/<path:path>')
def monitor(path):
    return send_from_directory(os.getcwd() + '/monitor/build', path)


if __name__ == '__main__':
    app.run(host='0.0.0.0')

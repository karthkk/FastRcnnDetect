import logging
import numpy as np
import tornado.web
import tornado

import tensorflow as tf
import _init_paths


from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
import json
import cv2

import cPickle

CLASSES =('__background__', 
                            "box",
                            "gum",
                            "marker",
                            "pen",
                            "postit",
                            "scissors",
                            "tape",
                            "usb"
                             )

class Detector():

    def __init__(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        gpu_id = 1
        demo_net = "VGGnet_test"
        model = "/data/code/Faster-RCNN_TF/output/faster_rcnn_end2end/office_supplies/VGGnet_fast_rcnn_iter_200.ckpt" 
        self.net = get_network(demo_net)
        saver = tf.train.Saver()
        saver.restore(self.sess, model)
     

    def detect(self, im):
        scores, boxes = im_detect(self.sess, self.net, im)
        return (scores, boxes)


def str2im(imstr):
    f1 = np.fromstring(imstr, dtype=np.uint8)
    return f1.reshape((480,640,3))

class DetectHandler(tornado.web.RequestHandler):

    def decode_argument(self, value, name=None):
        return value

    def post(self):
        global detector
        imstrjpg = self.get_argument('data', 'empty')
        if imstrjpg == 'emtpy':
            print 'EMPTY'
            return ""
        imstr = np.fromstring(imstrjpg, dtype=np.uint8)
        im = cv2.imdecode(imstr, cv2.CV_LOAD_IMAGE_UNCHANGED)
        scores, boxes = detector.detect(im)
        CONF_THRESH = 0.15
        NMS_THRESH = 0.08
        results = {}
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            results[cls] = dets

        self.write(cPickle.dumps(results))
        self.finish()

def make_app():
    return tornado.web.Application([
        (r"/detect/img", DetectHandler),
    ])

global detector
 
if __name__ == "__main__":
    detector = Detector()
    app = make_app()
    app.listen(6006)
    print("Starting server on 6006")
    tornado.ioloop.IOLoop.current().start()


    

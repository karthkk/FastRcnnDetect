import _init_paths
from networks import VGGnet_test
from fast_rcnn.test import im_detect

import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, name, classes, model_path, sess):
        self.name = name
        self.classes  = classes
        self.model_path = model_path
        self.sess = sess
        self.setup()

    def setup(self):
        scope_name = self.name
        with tf.variable_scope(scope_name):
            self.net = VGGnet_test(n_classes=len(self.classes))
            vars_in_net = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name)
            saver = tf.train.Saver(var_list=dict([(var.name.replace('%s/'%(scope_name) ,'').replace(':0',''), var)
                                              for var in vars_in_net]))
            saver.restore(self.sess, self.model_path)

    def detect(self, im):
        scores, boxes = im_detect(self.sess, self.net, im)
        NMS_THRESH = 0.08
        results = {}
        for cls_ind, cls in enumerate(self.classes[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            results[cls] = dets
        return results




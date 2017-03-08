import tensorflow as tf
from networks.network import Network


_feat_stride = [16,]
anchor_scales = [8, 16, 32]
arm_classes = 2
obj_classes = 9


class Robonet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info})
        self.trainable = trainable
        self.setup()


    def setup_subtask(self, subtask_name, n_classes):
        n = lambda x: subtask_name + '_' + x
        (self.feed('pool2').conv(3, 3, 256, 1, 1, name=n('conv3_1'))
         .conv(3, 3, 256, 1, 1, name=n('conv3_2'))
         .conv(3, 3, 256, 1, 1, name=n('conv3_3'))
         .max_pool(2, 2, 2, 2, padding='VALID', name=n('pool3'))
         .conv(3, 3, 512, 1, 1, name=n('conv4_1'))
         .conv(3, 3, 512, 1, 1, name=n('conv4_2'))
         .conv(3, 3, 512, 1, 1, name=n('conv4_3'))
         .max_pool(2, 2, 2, 2, padding='VALID', name=n('pool4'))
         .conv(3, 3, 512, 1, 1, name=n('conv5_1'))
         .conv(3, 3, 512, 1, 1, name=n('conv5_2'))
         .conv(3, 3, 512, 1, 1, name=n('conv5_3')))

        (self.feed(n('conv5_3'))
         .conv(3, 3, 512, 1, 1, name=n('rpn_conv/3x3'))
         .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name=n('rpn_cls_score')))

        (self.feed(n('rpn_conv/3x3'))
         .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name=n('rpn_bbox_pred')))

        (self.feed(n('rpn_cls_score'))
         .reshape_layer(2, name=n('rpn_cls_score_reshape'))
         .softmax(name=n('rpn_cls_prob')))

        (self.feed(n('rpn_cls_prob'))
         .reshape_layer(len(anchor_scales) * 3 * 2, name=n('rpn_cls_prob_reshape')))

        (self.feed(n('rpn_cls_prob_reshape'), n('rpn_bbox_pred'), 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TEST', name=n('rois')))

        (self.feed(n('conv5_3'), n('rois'))
         .roi_pool(7, 7, 1.0 / 16, name=n('pool_5'))
         .fc(4096, name=n('fc6'))
         .fc(4096, name=n('fc7'))
         .fc(n_classes, relu=False, name=n('cls_score'))
         .softmax(name=n('cls_prob')))

        (self.feed(n('fc7'))
         .fc(n_classes * 4, relu=False, name=n('bbox_pred')))

    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2'))
        self.setup_subtask("arm", arm_classes)
        self.setup_subtask('obj', obj_classes)



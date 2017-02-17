import tensorflow as tf
from networks.network import Network


#define

n_classes_obj = 9
n_classes_arm = 2

_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)


    def setup_subtask(self, subtask_name):
        n = lambda x: subtask_name + '_' + x
        (self.feed('pool2')
             .conv(3, 3, 256, 1, 1, name=n('conv3_1'))
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

        #========= RPN ============
        (self.feed(n('conv5_3'))
         .conv(3, 3, 512, 1, 1, name=n('rpn_conv/3x3'))
         .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name=n('rpn_cls_score')))

        (self.feed(n('rpn_cls_score'), 'gt_boxes', 'im_info', 'data')
         .anchor_target_layer(_feat_stride, anchor_scales, name=n('rpn-data')))

        # Loss of rpn_cls & rpn_boxes

        (self.feed(n('rpn_conv/3x3'))
         .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name=n('rpn_bbox_pred')))

        # ========= RoI Proposal ============
        (self.feed(n('rpn_cls_score'))
         .reshape_layer(2, name=n('rpn_cls_score_reshape'))
         .softmax(name=n('rpn_cls_prob')))

        (self.feed(n('rpn_cls_prob'))
         .reshape_layer(len(anchor_scales) * 3 * 2, name=n('rpn_cls_prob_reshape')))

        (self.feed(n('rpn_cls_prob_reshape'), n('rpn_bbox_pred'), 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name=n('rpn_rois')))

        (self.feed(n('rpn_rois'), 'gt_boxes')
         .proposal_target_layer(n_classes_obj, name=n('roi-data')))

        #========= RCNN ============
        (self.feed(n('conv5_3'), n('roi-data'))
             .roi_pool(7, 7, 1.0/16, name=n('pool_5'))
             .fc(4096, name=n('fc6'))
             .dropout(0.5, name=n('drop6'))
             .fc(4096, name=n('fc7'))
             .dropout(0.5, name=n('drop7'))
             .fc(n_classes_obj, relu=False, name=n('cls_score'))
             .softmax(name=n('cls_prob')))

        (self.feed(n('drop7'))
         .fc(n_classes_obj * 4, relu=False, name=n('bbox_pred')))



    def setup(self):

        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
             .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
             .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2'))

        self.setup_subtask('arm')
        self.setup_subtask('obj')





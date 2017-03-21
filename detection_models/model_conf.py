import _init_paths
import os
from model_containers import Model
model_output_folder = ''
this_dir = os.path.dirname(__file__)
model_location = this_dir + '/../output/faster_rcnn_end2end/%s'



def setup(sess):
    models = []
    models.append(Model("office_supplies", ('__background__',
                                         "box",
                                         "gum",
                                         "marker",
                                         "pen",
                                         "postit",
                                         "scissors",
                                         "tape",
                                         "usb"
    ), model_location%"office_supplies/VGGnet_fast_rcnn_iter_200.ckpt", sess))
    models.append(Model("arm", ('__background__',
                                         "arm"
    ), model_location%"armpos/VGGnet_fast_rcnn_iter_2000.ckpt", sess))
    return models


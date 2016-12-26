# Faster-RCNN_TF

This is an experimental Tensorflow implementation of Faster RCNN - a convnet for object detection with a region proposal network.
For details about R-CNN please refer to the paper [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/pdf/1506.01497v3.pdf) by Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.


### Requirements: software

1. Requirements for Tensorflow (see: [Tensorflow](https://www.tensorflow.org/)) - Tested r.011

2. Python packages needed to run this repo: `cython`, `python-opencv`, `easydict`

### Requirements: hardware

1. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/bigsnarfdude/Faster-RCNN_TF.git
  ```

2. Build the Cython modules
    ```Shell
    cd Faster-RCNN_TF/lib
    make
    ```

### Quickstart Demo and validate everything is installed correctly

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

You are going to need to download a pretrained model that was trained on PASCAL VOC 2007 dataset. Download from either of these locations and put into the Faster-RCNN_TF directory [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbZ0EzeUlHcGFIVWM) [[Dropbox]](https://www.dropbox.com/s/cfz3blmtmwj6bdh/VGGnet_fast_rcnn_iter_70000.ckpt?dl=0)

Get pretrained model
```Shell
wget https://www.dropbox.com/s/cfz3blmtmwj6bdh/VGGnet_fast_rcnn_iter_70000.ckpt?dl=0#
mv VGGnet_fast_rcnn_iter_70000.ckpt?dl=0 VGGnet_fast_rcnn_iter_70000.ckpt
```

To run the demo
```Shell
cd Faster-RCNN_TF/tools

python demo.py --model /home/ubuntu/Faster-RCNN_TF/VGGnet_fast_rcnn_iter_70000.ckpt --gpu 0

```

To run the demo IPython Notebook:

https://github.com/bigsnarfdude/Faster-RCNN_TF/blob/master/tools/demo_faster_rcnn_pedestrian_detector.ipynb



The demo.py will load the above pretrained Faster-RCNN_TF network that includes (VGG_ImageNet base and PASCAL 2007 training) and perform detection on each image in the demo.py list. demo.py is located in the tools folder from the project root. The gpu flag will target the first GPU on your box. You can add images into the data folder and run the demo.py against any of your custom photos once you refactor the image list. https://github.com/bigsnarfdude/Faster-RCNN_TF/blob/master/tools/demo.py

### Training Model from scratch for fun
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    
5. Download pre-trained ImageNet models

   Download the pre-trained ImageNet models [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)
   
   	```Shell
    mv VGG_imagenet.npy $FRCN_ROOT/data/pretrain_model/VGG_imagenet.npy
    ```

6. Run script to train and test model
	```Shell
	cd $FRCN_ROOT
	./experiments/scripts/faster_rcnn_end2end.sh GPU_ID VGG16 pascal_voc
	./experiments/scripts/faster_rcnn_end2end.sh $DEVICE $DEVICE_ID VGG16 pascal_voc
	```

### The result of testing on PASCAL VOC 2007 

| Classes       | AP     |
|-------------|--------|
| aeroplane   | 0.698 |
| bicycle     | 0.788 |
| bird        | 0.657 |
| boat        | 0.565 |
| bottle      | 0.478 |
| bus         | 0.762 |
| car         | 0.797 |
| cat         | 0.793 |
| chair       | 0.479 |
| cow         | 0.724 |
| diningtable | 0.648 |
| dog         | 0.803 |
| horse       | 0.797 |
| motorbike   | 0.732 |
| person      | 0.770 |
| pottedplant | 0.384 |
| sheep       | 0.664 |
| sofa        | 0.650 |
| train       | 0.766 |
| tvmonitor   | 0.666 |
| mAP        | 0.681 |


###References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[A tensorflow implementation of SubCNN (working progress)](https://github.com/yuxng/SubCNN_TF)

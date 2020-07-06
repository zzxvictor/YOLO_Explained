import tensorflow as tf
import numpy as np
import os
from xml.etree import ElementTree
from abc import ABC, abstractmethod

# data loader abstract class


class Loader(ABC):
    """
    parameters:
        imgDir: the directory containing images
        annotDir: the directory containing annotation files
        anchorPath: the path to the anchor file
        classNamePath: the path to the class names text file
    """
    def __init__(self, imgDir, annotDir=None, anchorPath=None, classNamePath=None):
        self.imgDir = imgDir
        self.annotDir = annotDir
        self.anchorPath = anchorPath
        self.classNamePath = classNamePath
        self.anchors = None
        self.classList = None
        if self.anchorPath:
            self.anchors = self._anchorReader(self.anchorPath)
            self.anchors = np.array(self.anchors).reshape(-1, 2)
            self.anchors = tf.convert_to_tensor(self.anchors, dtype=tf.float32)
        if self.classNamePath:
            self.classList = self._classReader(self.classNamePath)

    """
    call this function to load data into memory. This is an abstract class that needs different implementation 
    when using different data sets. 
    """
    @abstractmethod
    def loadData(self, batchSize=4, imgShape=None, repeat=False, shuffle=False):
        ...

    """
    preprocessing steps needed to tranform the data from text file to desired yolo format. 
    """
    @abstractmethod
    def _preprocess(self):
        ...

    """
    load an image from disk 
    parameters:
        imgPath: string tensor, path the one image 
    returns:
        img: tensor, of shape (img h, img w)
        imgShape, tensor, of shape (3, ) the shape of the image
    """

    def _decodeJpg(self, imgPath):
        img = tf.io.read_file(imgPath)
        # data normalized between 0 and 1
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img, tf.shape(img)

    """
    load an xml file from disk 
    parameters:
        xmlPath: string tensor, path the one xml file  
    returns:
        boxList: numpy array, of shape (num box, 5) 
    """

    def _decodeXml(self, xmlPath):
        # xmlPath should be a tensor
        xml = open(xmlPath.numpy().decode('utf-8'), 'r')
        tree = ElementTree.parse(xml)
        size = tree.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        bboxList = []
        for obj in tree.findall('object'):
            idx = self.classList.index(obj.find('name').text)
            bbox = obj.find('bndbox')
            xMin, yMin = int(bbox.find('xmin').text), int(bbox.find('ymin').text)
            xMax, yMax = int(bbox.find('xmax').text), int(bbox.find('ymax').text)
            bboxList.append([idx, xMin / w, yMin / h, xMax / w, yMax / h])
        xml.close()
        return np.array(bboxList).reshape(-1, 5)

    """
    load an txt file from disk 
    parameters:
        txtPath: string tensor, path the one text file 
    returns:
        boxList: numpy array, of shape (num box, 5) 
    """

    def _decodeTxt(self, txtPath):
        file = open(txtPath.numpy().decode('utf-8'), 'r')
        content = file.read()
        content = content.strip('\n').split('\n')
        content = [list(map(float, line.split(' '))) for line in content if line != '']
        return np.array(content).reshape(-1, 5)


    """
    get the id of the best matching anchor for each ground truth box
    parameters:
        labels: tensor, of shape (conv H, convH, 4)
        anchors: tensor, of shape (num anchor, 2)
    returns:
        bestIdx: tensor, (conv H, convH)
    """

    def _getAnchorIdx(self, label, anchors):
        anchors = tf.cast(anchors, label.dtype)
        boxWH = label[..., 2:4]
        boxWH = tf.tile(boxWH, multiples=[1, anchors.shape[0]])
        # duplicate the boxes so that we can calculate
        # its intersection with each anchor box
        boxWH = tf.reshape(boxWH, (-1, anchors.shape[0], 2))
        # boxWH of shape (num box, num anchor, 2)
        overlap = tf.math.minimum(boxWH, anchors)
        overlap = overlap[..., 0] * overlap[..., 1]
        # overlap of shape (num box, num anchor)
        boxArea = boxWH[..., 0] * boxWH[..., 1]
        anchorArea = anchors[..., 0] * anchors[..., 1]
        iou = overlap / (boxArea + anchorArea - overlap)
        # find the highest IOU and the anchor index
        bestIdx = tf.argmax(iou, axis=-1)
        return bestIdx

    """
    read the text file that contains the list of class names. 
    """

    def _classReader(self, path):
        file = open(path, 'r')
        classList = [line.strip('\n') for line in file]
        file.close()
        return classList

    """
    read the text file that contains the list of anchors. 
    """

    def _anchorReader(self, path):
        file = open(path, 'r')
        anchors = file.readline().strip('\n')
        anchors = anchors.split(',')
        # str to float
        anchors = [float(val.strip()) for val in anchors]
        return anchors


"""
data loader designed for Pascal style data set (each image has a XML annotation file with the same name)
"""


class LoadPascal(Loader):
    """
    load the pascal data set
    parameters:
        batchSize: number of images per batch
        imgShape: None if you don't want to reshape the input images (code might not work because images need to be
            divisible by 32). Tuple with value (new height, new width, 3)
        repeat: repeat the data endlessly (require manual termination) if set to True
        shuffle: shuffle the images if set to True
        imgOnly: does not load annotations if set to False (for evaluation)
    return:
        data: high performance tf.data.dataset instance. supports data processing while loading,
        overlapping loading with training.
    """
    def loadData(self, batchSize=4, imgShape=None, repeat=False, shuffle=False, imgOnly=False):
        # sanity checks
        if imgShape is not None and ((len(imgShape) < 3 or imgShape[-1] != 3) or\
                                     (imgShape[0] % 32 != 0 or imgShape[1] % 32 != 0)):
            raise ValueError('image shape is illegitimate. Must be divisible by 32')
        if imgShape is None and batchSize != 1:
            raise ValueError('batch size must be 1 if img shape is not provided (imgs have different shapes)')
        if imgShape is None and not imgOnly:
            raise ValueError('training images must have the same shape, imgShape parameter is needed' +
                             'for instance, imgShape=(416, 416, 3)')

        # get all files under the image folder
        imgs = [file.split('.')[0] for file in os.listdir(self.imgDir) if file.endswith('.jpg')]
        # shuffle the images before loading reduces shuffling work
        if shuffle:
            np.random.shuffle(imgs)
        imgs = tf.data.Dataset.from_tensor_slices(imgs)
        data = imgs.map(lambda x: self._preprocess(x, imgShape, imgOnly),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.batch(batchSize)
        if repeat:
            data = data.repeat()
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return data

    def _preprocess(self, imgName, imgShape, imgOnly=False):
        imgPath = tf.strings.join([self.imgDir, '/', imgName, '.jpg'])

        # load in an image
        img, oriShape = self._decodeJpg(imgPath)
        if imgShape is not None:  # resize an image if imgShape is provided
            img = tf.image.resize(img, imgShape[0: 2])
        else:
            imgShape = oriShape
        if imgOnly:
            return img, imgPath

        # load in the Pascal annotation
        annotPath = tf.strings.join([self.annotDir, '/', imgName, '.xml'])
        # _decodeXML is not a native tf function, must be wrapped by py_function
        # so that it can be added to a tf graph
        label = tf.py_function(self._decodeXml, inp=[annotPath], Tout=[tf.float32])
        label = tf.squeeze(label, axis=0)
        # label has shape [num box, 5] right now. We can't batch label of different imgs together
        # because they might have different number of boxes
        # muse convert them ito yolo format
        label = self._yoloFormat(label, imgShape)
        # label has shape (convH, convW, num anchors, 6)
        # label[i,j,k,0] = 0/1, indicates if a box is present
        return img, label, imgPath

    """
    convert the annotation to yolo format so that we can batch the dataset
    """

    def _yoloFormat(self, label, imgShape):
        convH, convW = imgShape[0] // 32, imgShape[1] // 32

        # from box coordinates to box height/width and center location
        boxXY = 0.5 * (label[:, 1:3] + label[:, 3:5])
        boxWH = label[:, 3:5] - label[:, 1:3]
        # reorganize the layout and scale the boxes
        label = tf.concat([boxXY, boxWH, label[:, 0:1]], axis=1)
        scale = tf.stack([convH, convW, convH, convW, 1], axis=-1)
        scale = tf.cast(scale, label.dtype)
        label = label * scale
        # get the best anchor index for each bbox
        idx = self._getAnchorIdx(label, self.anchors)
        Y = tf.math.floor(label[:, 1])  # Y
        X = tf.math.floor(label[:, 0])  # X
        numAnchor = self.anchors.shape[0]

        # modify masks in numpy because tensor obj cannot be modified
        def getMasks(X, Y, idx, anchors, label, convH, convW):
            labels = np.zeros((convH, convW, numAnchor, 6))
            for i, (y, x, anchor) in enumerate(zip(Y, X, idx)):
                xInt, yInt, aInt = int(x), int(y), int(anchor)
                # to understand this step, please refer to
                # https://arxiv.org/pdf/1612.08242.pdf page 3
                labels[yInt, xInt, aInt] = np.array([1,  # indicate an object appears in this box
                                                     # box center coordinates
                                                     label[i][0] - x,
                                                     label[i][1] - y,
                                                     # box width w.r.t to the anchor box
                                                     label[i][2] / anchors[aInt][0],
                                                     label[i][3] / anchors[aInt][1],
                                                     label[i][-1]])  # object class id
            return labels

        # getMasks is not a native tf function, must be wrapped by py_function
        # so that it can be added to a tf graph.
        # we use a numpy function here because tensors cannot be modified in place (but np array can)
        labels = tf.py_function(getMasks, inp=[X, Y, idx, self.anchors, label, convH, convW], Tout=[tf.float32])

        return tf.squeeze(labels)


"""
convert the labels generated by the loader functions back to a list of boxes (for sanity check or if you want to visualize)
the training data set
parameters:
    labels: tensor, of shape (conv H, conv W, num anchors, 6). generated by the loading function 
    convShape: the height and width of the label
returns:
    labels: a list of boxes 
"""


def label2Box(labels, convShape, anchors):
    mask = tf.squeeze(labels[..., 0:1])
    labels = tf.squeeze(labels[..., 1:])

    hIndex = tf.reshape(tf.range(start=0, limit=convShape[0]), (convShape[0], 1))
    hIndex = tf.tile(hIndex, [1, convShape[1]])  # expand in the height direction
    wIndex = tf.reshape(tf.range(start=0, limit=convShape[1]), (1, convShape[1]))
    wIndex = tf.tile(wIndex, [convShape[0], 1])  # expand in the width direction
    idx = tf.stack([wIndex, hIndex], axis=-1)
    idx = tf.reshape(idx, shape=(convShape[0], convShape[1], 1, 2))
    idx = tf.cast(idx, labels.dtype)

    boxXY = (labels[..., 0:2] + idx) / convShape
    boxWH = (labels[..., 2:4]) * anchors / convShape

    classId = labels[..., 4:5]

    labels = tf.concat([classId, boxXY - boxWH / 2, boxXY + boxWH / 2], axis=-1)
    labels = tf.boolean_mask(labels, mask)

    return labels
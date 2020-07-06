import tensorflow as tf

"""
convert raw yolo output to a list of boxes
parameters:
    featureMap: tensor, (None, conv H, conv W, num anchors * (num class + 5))
    anchors:  tensor, (num anchors, 2) [[w, h],[w, h],...]
    numClass: int, the number of classes 
    moxBox: int, the maximum number of boxes returned 
    scoreThresh: float, used to filter out low confidence boxes 
    iouThresh: float, used to filter out boxes that are significantly overlapped 
returns:
    generator of values: 
        boxes: list of boxes for one image [[left top right bottom], [], []]
        scores: list of scores for those boxes [score1, score2, ...]
        classes: class prediction [pred1, pred2, ...]
    
"""


def raw2Box(featureMap, anchors, numClass, maxBox=20, scoreThresh=0.5, iouThresh=0.5):
    batchXY, batchWH, batchScore, batchProb = extractInfo(featureMap, anchors, numClass)
    batchLoc = getBoxLoc(batchXY, batchWH)
    batchLoc = scaleBox(batchLoc, (32, 32))
    for boxLoc, objScore, classProb in zip(batchLoc, batchScore, batchProb):
        boxes, scores, classes = filterBox(boxLoc, objScore, classProb, scoreThresh)
        boxes, scores, classes = nonMaxSuppress(boxes, scores, classes, maxBox, iouThresh)
        yield boxes, scores, classes


"""
generate offsets for a given shape 
parameters:
    shape: tuple of length 2, [conv height, conv weidth]
returns:
    offset: tensor of shape [1. conv H, conv W, 1, 2]. 
        Essentially the Cx and Cy mentioned in tutorial 1 option 2, used to convert box coordinates 
        from w.r.t to grids to w.r.t to the feature map 
"""


def getOffset(shape):
    hIndex = tf.reshape(tf.range(start=0, limit=shape[0]), (shape[0], 1))
    hIndex = tf.tile(hIndex, [1, shape[1]])  # expand in the height direction
    wIndex = tf.reshape(tf.range(start=0, limit=shape[1]), (1, shape[1]))
    wIndex = tf.tile(wIndex, [shape[0], 1])  # expand in the width direction
    idx = tf.stack([wIndex, hIndex], axis=-1)
    idx = tf.reshape(idx, shape=(1, *shape, 1, 2))
    return idx


"""
extract bounding box information from raw model outputs 
parameters:
    modelOuput: tensor, of shape (None, conv H, conv W, num anchors * (num class + 5))
    anchors: tensor, of shape (num anchors, 2)
    num class: int, number of classes 
returns:
    boxXY: tensor, of shape (None, conv H, conv W, num anchors, 2). The coordinates of box centers
    boxWH: tensor, of shape (None, conv H, conv W, num anchors, 2). The width and height of boxes
    objScore: tensor, of shae (None, conv H, conv W, num anchors, 1). The obj score of each anchor box 
    classProb: tensor, of shape (None, conv H, conv W, num anchors, num class). The class probabilities 
"""


def extractInfo(modelOutput, anchors, numClass):
    featureDim = modelOutput.shape
    numAnchor = anchors.shape[0]
    modelOutput = tf.reshape(modelOutput, shape=(-1, featureDim[1], featureDim[2], numAnchor, numClass + 5))
    imageShape = featureDim[1:3]
    # create anchor index
    idx = getOffset(imageShape)
    idx = tf.cast(idx, modelOutput.dtype)
    # extract box coordinates
    # please refer to the original paper for better understanding
    boxXY = tf.nn.sigmoid(modelOutput[..., :2])  # x and y between 0 and 1
    boxWH = tf.math.exp(modelOutput[..., 2:4])  # relative w and h
    objScore = tf.nn.sigmoid(modelOutput[..., 4:5])  # objectiveness score
    classProb = tf.nn.softmax(modelOutput[..., 5:])  # probability of classes
    # offset by grid coordinates
    anchors = tf.cast(tf.reshape(anchors, (1, 1, 1, numAnchor, 2)), idx.dtype)
    boxXY = (boxXY + idx)  # boxXY becomes coordinates w.r.t to the top left corner
    boxWH = boxWH * anchors
    return boxXY, boxWH, objScore, classProb


"""
convert the box center and weight to top left and bottom right corners 
parameters: 
    boxXY: tensor, of shape (None, conv H, conv W, num anchors, 2). The coordinates of box centers
    boxWH: tensor, of shape (None, conv H, conv W, num anchors, 2). The width and height of boxes
returns:
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). The top left and bottom right coordinates 
"""


def getBoxLoc(boxXY, boxWH):
    topLeft = boxXY - boxWH / 2  # top left
    bottomRight = boxXY + boxWH / 2  # bottom right
    # the last dimension is (x1, y1, x2, y2)
    # top left means it is closer to (0,0) in the image, which is the top-left corner
    # if displayed by matplotlib 
    return tf.concat([topLeft, bottomRight], axis=-1)


"""
filter out boxes with low object score 
parameters:
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). The top left and bottom right coordinates 
    objScore: tensor, of shae (None, conv H, conv W, num anchors, 1). The obj score of each anchor box 
    classProb: tensor, of shape (None, conv H, conv W, num anchors, num class). The class probabilities 
    scoreThresh: filter threshold
returns 
    boxes: list of boxes, [box1, box2, ...]
    objScore: list of scores, [score1, score2, ...]
    classProb: list of class probability, [prob1, prob2, ...]
"""


def filterBox(boxLoc, objScore, classProb, scoreThresh=0.5):
    boxScore = objScore * classProb  # (None, B1, B2, S, NCLASS)
    boxClass = tf.argmax(boxScore, axis=-1)  # shape = (None, S, S, B)
    boxScore = tf.math.reduce_max(boxScore, axis=-1)  # shape = (None, S, S, B)
    mask = boxScore >= scoreThresh
    # filter out low-confidence boxes
    boxes = tf.boolean_mask(boxLoc, mask)
    scores = tf.boolean_mask(boxScore, mask)
    classes = tf.boolean_mask(boxClass, mask)
    return boxes, scores, classes


"""
scale the boxes to full image scale 
parameters: 
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). The top left and bottom right coordinates 
    scale: tuple, value = (32, 32). This is because conv dim = img dim // 32 
returns:
    boxLoc: tensor, of shape (None, conv H, conv W, num anchors, 4). scaled version of boxLoc
"""


def scaleBox(boxLoc, scale):
    height, width = scale[0], scale[1]
    shape = tf.stack([height, width, height, width])
    shape = tf.reshape(shape, [1, 4])
    shape = tf.cast(shape, boxLoc.dtype)
    return boxLoc * shape


"""
filter out boxes that are significantly overlapped. Only preserve the box with the highest score 
parameters:
    boxLoc: list of tensor, [box1, box2, ...]
    objScore: lit of tensor, [score1, score 2]
    iouThresh: float, filter threshold 
returns:
    boxLoc: list of tensor, [box1, box2, ...] boxes left 
    objScore: lit of tensor, [score1, score 2] their scores 
"""


def nonMaxSuppress(boxLoc, score, classPredict, maxBox=20, iouThresh=0.5):
    idx = tf.image.non_max_suppression(boxLoc, score, maxBox, iou_threshold=iouThresh)
    boxLoc = tf.gather(boxLoc, idx)
    score = tf.gather(score, idx)
    classPredict = tf.gather(classPredict, idx)
    return boxLoc, score, classPredict


"""
calculate the Intersect Over Union of two outputs 
    box1: tensor, of shape (None, convH, convW, num anchors, 4). 
        box[i, j, k, l, :] is a box of value [center x, center y, w, h]
    box2: tensor, of shape (None, convH, convW, num anchors, 4).
returns:
    iou: tensor, of shape (None, convH, convW, num anchors). The iou of boxes
    iou[i, j, k, l] is the iou of box1[i, j, k, l, :] and box2[i, j, k, l, :]
"""


def getIOU(box1, box2):
    mini = tf.math.maximum(box1[..., 0:2] - box1[..., 2:4] / 2, box2[..., 0:2] - box2[..., 2:4] / 2)
    maxi = tf.math.minimum(box1[..., 0:2] + box1[..., 2:4] / 2, box2[..., 0:2] + box2[..., 2:4] / 2)
    interWH = tf.math.maximum(maxi - mini, 0)
    interArea = interWH[..., 0] * interWH[..., 1]
    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    return interArea / (area1 + area2 - interArea)


"""
the loss function of yolo. 
parameters: 
    features: tensor, of shape (None, convH, convW, num anchor * (num class + 5) ) = y predicted 
    labels: tensor, of shape (None, convH, convW, num anchor * (num class + 5) ) = y true  
    anchors: tensor, of shape (num anchor, 2)
    hyperparam: dictionary, contains loss hyperparameters. For more information please read tutorial 5.
returns:
    loss: float, yolo loss 
    [object loss, classification loss, localization loss]: each has shape (batch size,). 
        offers insight into the composition of the loss value. Use for diagnosis
"""


def yoloLoss(features, labels, anchors, hyperparam={'local': 5.0, 'obj': 5.0, 'nonObj': 0.5, 'iouThresh': 0.5}):
    # get hyperparameters for the loss function
    localCoef, nonObjCoef, objCoef = hyperparam['local'], hyperparam['nonObj'], hyperparam['obj']
    fShape = tf.shape(features)
    # numAnchor = labels.shape[-2] # label has shape (None, convH, convW, num anchor, 6)
    numAnchor = anchors.shape[0]
    numClass = fShape[-1] // numAnchor - 5
    features = tf.reshape(features, shape=[fShape[0], fShape[1], fShape[2], numAnchor, numClass + 5])

    # convert raw features into box coordinates w.r.t to each square in the feature map
    # please refer to https://arxiv.org/pdf/1612.08242.pdf page 3
    offset = tf.cast(getOffset([fShape[1], fShape[2]]), features.dtype)
    boxXY = tf.nn.sigmoid(features[..., :2]) + offset
    boxWH = tf.math.exp(features[..., 2:4]) * anchors
    objScore = tf.nn.sigmoid(features[..., 4:5])
    classProb = tf.nn.softmax(features[..., 5:])
    # yolo loss is defined as
    # W1 * localization loss + W2 * confidence loss + classification loss
    # https://arxiv.org/abs/1506.02640 page 4

    # calculate the classification loss
    mask = labels[..., 0:1]  # signal if an object appears in a given location
    labels = labels[..., 1:]
    gtClasses = tf.cast(labels[..., 4], tf.int32)
    gtClasses = tf.one_hot(gtClasses, depth=numClass)
    classLoss = tf.square(classProb - gtClasses) * mask
    classLoss = tf.math.reduce_sum(classLoss, axis=[1, 2, 3, 4])

    # calculate the localization loss
    # coordinates w.r.t to each square in the feature map
    gtXY, gtWH = labels[..., 0:2] + offset, (labels[..., 2:4] * anchors)
    localLoss = tf.square(gtXY - boxXY) + tf.square(tf.sqrt(gtWH) - tf.sqrt(boxWH))
    localLoss *= mask
    localLoss = localCoef * tf.math.reduce_sum(localLoss, axis=[1, 2, 3, 4])

    # calculate the detection loss
    iou = getIOU(tf.concat([boxXY, boxWH], axis=-1), tf.concat([gtXY, gtWH], axis=-1))
    bestIOU = tf.math.reduce_max(iou, axis=-1, keepdims=True)
    nonObj = bestIOU < hyperparam['iouThresh']
    nonObj = tf.cast(tf.expand_dims(nonObj, axis=-1), mask.dtype) * (1.0 - mask) * nonObjCoef
    nonObjLoss = nonObj * tf.square(0 - objScore)
    objLoss = mask * tf.square(objScore - tf.expand_dims(iou, axis=-1)) * objCoef

    objLoss = tf.math.reduce_sum(nonObjLoss + objLoss, axis=[1, 2, 3, 4])
    # overall loss
    return tf.math.reduce_mean(objLoss + localLoss + classLoss), [objLoss, localLoss, classLoss]
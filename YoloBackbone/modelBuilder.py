import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import re


class YOLO2:
    """
    params:
        configPath: str, path to the yolo config file
        weightPath: str, path to the pre-trained weight file. Leave blank if you want a random model
        inputShape: tuple(int, int, int), the input size of the image. You can use (None, None, 3) to go fully convolutional
    """
    def __init__(self, configPath, weightPath=None, inputShape=(608, 608, 3), transfer=False):
        # map layer keywords to corresponding handlers
        self.keywords = {'net': self._netHandler,
                         'convolutional': self._convHandler,
                         'maxpool': self._poolHandler,
                         'reorg': self._reorgionHandler,
                         'route': self._routHandler,
                         'region': self._regionHandler}
        if inputShape is None:
            inputShape = (None, None, 3)

        self.layers = [tfk.layers.Input(shape=inputShape, name='input_0')]
        self.configPath = configPath
        # if not provided, model weights are randomly initialized
        self.weightPath = weightPath
        if weightPath:
            self.FLOAT32 = 4
            self.INT32 = 4
            self.darknet = open(weightPath, 'rb')
            # pop the first 4 int as they are GNU compiler headers
            # refer to https: // github.com / pjreddie / darknet / blob / master / src / parser.c
            # refer to https://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html
            major, minor, revision = np.ndarray(shape=(3,), dtype='int32',
                                                buffer=self.darknet.read(3 * self.INT32))
            if (major * 10 + minor) >=2 and major < 1000 and minor < 1000:
                seen = np.ndarray(shape=(1,), dtype='int64', buffer=self.darknet.read(8))
            else:
                seen = np.ndarray(shape=(1,), dtype='int32', buffer=self.darknet.read(4))
            #print('GNU compiler config: ', major, minor, revision, seen)
        else:
            print('No weights were loaded, you got an randomly initialized model')
        self.transfer = transfer
    """
    call this function to build the YOLO v2 model and load weights if a pre-trained file is provided
    parameters:
        silence: print out the model summary if set true 
    returns:
        model: tf keras mdoel 
        anchors: list of float, anchor shapes defined in the configuration file 
        classNum: int, the number of classes defined in the config file 
    """
    def buildModel(self, silence=False):
        config = open(self.configPath, 'r').read().strip('\n')
        if self.transfer: # transfer learning, get darknet weights only
            darknet = self._parser(re.split('(###+)', config)[0])
            print('only darknet weights are loaded!')
            for line in darknet:
                self._extractor(line)
            self.weightPath = None # do not read weights after darkNet                 
            yolo = self._parser(re.split('(###+)', config)[2])
        else:
            yolo = self._parser(config)

        for line in yolo:
            self._extractor(line)
        # wrap up into a tf keras model
        model = tfk.Model(inputs=self.layers[0], outputs=self.layers[-1])
        if not silence:
            print(model.summary())
        # pre-determined anchor boxes are essential for the performance
        # typically the anchor boxes are determined by K means
        return model, self.anchors, self.classNum
# helper functions
    """
    parse the config file by newline and [ ] sign
    parameters:
        string: str, contains the configuration details
    return:
        list of strings, each string is a block in the config file
    """
    def _parser(self, string):
        pattern = r'\n+(?=\[.+\])'
        parsed = re.split(pattern, string)
        return [string for string in parsed if len(string) > 0]
    """
    extract the name of the block and call the appropriate handler
    params:
        line: str, for instance '[convolution]\nsize=1\nfilters=32'
    return:
        None
    """
    def _extractor(self, line):
        keyword = re.search(r'\w+', line).group(0)
        # call the handler
        self.keywords[keyword](line)
    """
    extract the value of the key argument. For instance, the function 
    extracts 1 from 'strides = 1'.
    params:
        key: str, the name of the field 
        line: str, string containing the key and the value
    returns:
        value: int
    """
    def _argExtractor(self, key, line):
        pattern = r'(?<=({}=)).+'.format(key)
        result = re.search(pattern, line)
        if result:
            return result.group()
        else:
            return 0
# handler functions
    """
    read the config file and determine necessary model hyperparameters 
    params:
        args: str
    """
    def _netHandler(self, args):
        self.decay = float(self._argExtractor('decay', args))

    def _regionHandler(self, args):
        self.anchors = self._argExtractor('anchors ', args).split(',')
        self.anchors = [float(val) for val in self.anchors]
        self.classNum = int(self._argExtractor('classes', args))
    """
    read the config file and build skip connections or concatenation layers
    params:
        args: str, specifications of the route 
    """
    def _routHandler(self, args):
        loc = self._argExtractor('layers', args).split(',')
        loc = list(map(lambda x: int(x), loc))

        if len(loc) == 1:
            self.layers.append(self.layers[loc[0]])
        else:
            layerList = [self.layers[i] for i in loc]
            concat = tfk.layers.concatenate(layerList,
                                            name='concate_' + str(len(self.layers)))
            self.layers.append(concat)
    """
    read the config file and build a space to depth layer 
    params:
        args: str, specifications of the layer 
    """
    def _reorgionHandler(self, args):
        stride = int(self._argExtractor('stride', args))
        def space2DepthWrapper(x):
            return tf.nn.space_to_depth(x, block_size=stride,
                                        name='space2Depth_' + str(len(self.layers)))
        self.layers.append(space2DepthWrapper(self.layers[-1]))
    """
    read the config file and build a max pooling layer 
    params:
        args: str, specifications of the layer 
    """
    def _poolHandler(self, args):
        size = int(self._argExtractor('size', args))
        stride = int(self._argExtractor('stride', args))
        pooling = tfk.layers.MaxPool2D(padding='same', pool_size=size,
                                       strides=stride,
                                       name='maxPool_' + str(len(self.layers)))
        self.layers.append(pooling(self.layers[-1]))
    """
    read the config file and the weight file if it is provided; build a conv layer
    params:
        args: str, specifications of the layer 
    """
    def _convHandler(self, args):
        filters = int(self._argExtractor('filters', args))
        size = int(self._argExtractor('size', args))
        stride = int(self._argExtractor('stride', args))
        pad = int(self._argExtractor('pad', args))
        pad = 'same' if pad == 1 else 'valid'
        batchNorm = int(self._argExtractor('batch_normalize', args))
        batchNorm = True if batchNorm == 1 else False
        activation = self._argExtractor('activation', args)
        # assume channel last
        inChannel = self.layers[-1].shape[-1]

        if self.weightPath:
            # read in weights
            if not batchNorm:  # bias only
                biasWeights = np.ndarray(shape=(filters,),
                                         dtype='float32',
                                         buffer=self.darknet.read(filters * self.FLOAT32))
            else:  # batch norm only
                bnWeights = np.ndarray(shape=(4, filters),
                                       dtype='float32',
                                       buffer=self.darknet.read(4 * filters * self.FLOAT32))
                bnWeights = [bnWeights[1], bnWeights[0], bnWeights[2], bnWeights[3]]
                bnArg = {'weights': bnWeights}

            darkNetShape = [filters, inChannel, size, size]
            convWeights = np.ndarray(shape=darkNetShape, dtype='float32',
                                     buffer=self.darknet.read(np.product(darkNetShape) * self.FLOAT32))
            convWeights = np.transpose(convWeights, [2, 3, 1, 0])  # channel first to channle last
            convWeights = [convWeights] if batchNorm else [convWeights, biasWeights]
            convArg = {'weights': convWeights}
        else:
            # if no weights are specified, let Keras handle the initialization
            convArg = {}
            bnArg ={}

        # create layers
        output = tfk.layers.Conv2D(filters=filters, kernel_size=size,
                                   strides=stride, padding=pad,
                                   name='conv_' + str(len(self.layers)),
                                   kernel_regularizer=tfk.regularizers.l2(self.decay),
                                   use_bias=not batchNorm,
                                   **convArg)(self.layers[-1])

        if batchNorm:
            output = tfk.layers.BatchNormalization(name='batchNorm_' + str(len(self.layers)),
                                                   **bnArg)(output)
        if activation == 'leaky':
            output = tfk.layers.LeakyReLU(alpha=0.1,
                                          name='lReLu_' + str(len(self.layers)))(output)
        self.layers.append(output)

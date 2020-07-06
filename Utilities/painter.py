import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class Visualizer:
    def __init__(self, classNames=[]):
        if len(classNames) == 0:
            self.colorMap = None
            self.classNames = []
        else:
            # assign random color to each class
            self.colorMap = colorMap = [(1, np.random.rand(), np.random.rand()) for i in range(len(classNames))]
            self.classNames = classNames

    def drawBox(self, image, boxes, scores=[], classes=[]):
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(6, 6)
        ax.set_axis_off()
        # sanity check
        if len(classes) > 0 and np.max(classes) > len(self.colorMap) - 1:
            print('make sure the class name list provided at initialization matches the class predictions')
            return None
        ax.imshow(image, aspect='auto')
        for i, box in enumerate(boxes):
            coor = [box[0], box[1]]
            #coor[0] = np.clip(coor[0], a_min=0, a_max=image.shape[1])
            #coor[1] = np.clip(coor[1], a_min=0, a_max=image.shape[0])
            w, h = box[2] - box[0], box[3] - box[1]
            #w = np.clip(w, a_min=0, a_max=image.shape[1])
            #h = np.clip(h, a_min=0, a_max=image.shape[0])
            if self.colorMap and len(classes) > 0:
                rec = patches.Rectangle(coor, w, h, fill=False, color=self.colorMap[classes[i]])
            else:
                rec = patches.Rectangle(coor, w, h, fill=False)
            ax.add_patch(rec)
            info = ''
            if len(classes) > 0 and self.classNames:
                info += self.classNames[classes[i]]
            if len(scores) > 0 is not None:
                info += '{0:.2f}'.format(scores[i])
            if info == '':  # do not create text boxes
                continue
            if self.colorMap is None or len(classes) == 0:
                fill = dict(boxstyle='square,pad=0.2')
            else:
                fill = dict(boxstyle='square,pad=0.2', facecolor=self.colorMap[classes[i]])
            ax.text(*coor, s=info, bbox=fill)
        return fig
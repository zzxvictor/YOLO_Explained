import xml.etree.ElementTree as ET
import os

"""
convert Pascal xml labels to text files 
parameters:
    labelDir: directory that contains xmls 
    outDir: where you want to dump the text files 
    imgShape: the training shape you use (have to rescale ground truth for fair comparision)
returns:
    None 
"""


def labelConverter(labelDir, outDir, imgShape):
    files = os.listdir(labelDir)
    outFiles = ['/'.join([outDir, file.replace(".xml", ".txt")]) for file in files]
    inFiles = ['/'.join([labelDir, file]) for file in files]

    for inFile, outFile in zip(inFiles, outFiles):
        temp = []
        handle = open(inFile, "r")
        root = ET.parse(handle).getroot()
        dim = root.find('size')
        width = int(dim.find('width').text)
        height = int(dim.find('height').text)

        wScale = imgShape[1] / width
        hScale = imgShape[0] / height

        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = int(bndbox.find('xmin').text) * wScale
            top = int(bndbox.find('ymin').text) * hScale
            right = int(bndbox.find('xmax').text) * wScale
            bottom = int(bndbox.find('ymax').text) * hScale
            line = '{} {} {} {} {}'.format(obj_name, left, top, right, bottom)
            temp.append(line)
        handle.close()
        handle = open(outFile, 'w')
        for line in temp:
            handle.write(line + '\n')
        handle.close()


"""
convert the predicted boxes to text file 
parameters:
    boxes, box coordinates
    scores, scores of each boxes
    classIdx, the class prediction of each boxes
    name: the list of class names, used to convert class index to text 
    outDir: where you want to dump the text files 
returns:
    None 
"""


def detectionConverter(boxes, scores, classIdx, classList, name, outDir):
    res = []
    for box, score, idx in zip(boxes, scores, classIdx):
        left, top, right, bottom = box.numpy().ravel()
        className = classList[idx.numpy()]
        info = [className, score.numpy(), left, top, right, bottom]
        res.append(info)
    res = sorted(res, key=lambda x: x[1], reverse=True)
    name = name.decode("utf-8").split('/')[-1]
    outName = os.path.join(outDir, name.replace('.jpg', '.txt'))
    handle = open(outName, 'w')
    for box in res:
        className, vals = box[0:1], box[1:]
        vals = list(map(str, vals))
        record = ' '.join(className + vals)
        handle.write(record + '\n')
    handle.close()


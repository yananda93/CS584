import numpy as np 
import cv2
import time
import xml.etree.ElementTree as ET

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
annotations_path = 'TrainVal/VOCdevkit/VOC2011/Annotations/'
image_path = 'TrainVal/VOCdevkit/VOC2011/JPEGImages/'
image_name_path = 'TrainVal/VOCdevkit/VOC2011/ImageSets/Main/val.txt'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
IOU_THRESHOLD = 0.5

# get the validation set
def getFileNames(cls):
    path = image_name_path
    file_names = []

    with open(path) as f:
        rows = f.read().splitlines()  # read the file into a list
    for row in rows:
        file_names.append(row.split()[0])
    return file_names


# calculate IoU , box is represted as (centerx, centery, width, height)
def iou(bb1, bb2):
    # get intersection
    width1, height1 = bb1[2], bb1[3]
    width2, height2 = bb2[2], bb2[3]
    ll1 = bb1[0] - width1 / 2
    ll2 = bb2[0] - width2 / 2
    lr1 = bb1[0] + width1/ 2
    lr2 = bb2[0] + width2 / 2
    l1 = bb1[1] - height1 / 2
    l2 = bb2[1] - height2 / 2
    h1 = bb1[1] + height1 / 2
    h2 = bb2[1] + height2 / 2
    area1 = width1*height1
    area2 = width2*height2
    left = max(ll1, ll2)
    right = min(lr1, lr2)
    low = max(l1, l2)
    high = min(h1,h2)
    w = max(0,right - left)
    h = max(0, high - low)
    inter_area = w*h

    # get union
    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def mAP():
    pass

# The following 2 functions are modified from darknet: https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_name):
    boxes = []
    classIDs = []
    annotation_file = annotations_path + image_name +'.xml'
    in_file = open(annotation_file)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        classIDs.append(cls_id)
        boxes.append(bb)
        # out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return classIDs, boxes


def detect(darknet, layer, filename):
# Refered openCV tutorial https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    boxes = []
    confidences = []
    classIDs = []

    path = image_path + filename + '.jpg'
    img = cv2.imread(path)
    (H, W) = img.shape[:2]
    # create input blob 
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416,416), (0,0,0), True, crop=False)
    darknet.setInput(blob)
    start = time.time()
    layerOutputs = darknet.forward(layer)
    end = time.time()
    run_time =  (end - start)

    # get the bounding boxes and classes
    for output in layerOutputs:
        # a detection contains bounding boxe(centerx, centry, width, height), 
        # box confidence, and class condidence(4 + 1 + 20)
        for detection in output:
            # extract the class ID and confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE_THRESHOLD:
            
                boxes.append(detection[0:4])
                confidences.append(float(confidence))
                classIDs.append(classID)
    return run_time, classIDs, boxes, confidences


def evalutaion(cfg, weights, darknet, layer, filenames, c):
    predicts = []
    recall_levels = np.linspace(0.0,1.0, num=11)  
    num_ground_truth = 0 
    total_time  = 0 
    for file in filenames:
  
        run_time, classIDs, boxes, confidences = detect(darknet, layer, file)
        total_time += run_time
      #  idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
      
        true_classIDs, true_boxes = convert_annotation(file)
        for i in range(len(classIDs)):
            if c == classIDs[i]:
                highest_iou = 0
                TP = 0
                for true_cls, ture_box in zip(true_classIDs, true_boxes):
                    if classIDs[i] == true_cls:
                        num_ground_truth += 1
                        IOU = iou(ture_box, boxes[i])
                        if IOU > highest_iou:
                            highest_iou = IOU

                if  highest_iou >= IOU_THRESHOLD:  
                    TP = 1   

                predicts.append([confidences[i], TP])
    predicts.sort(reverse=True)  # sort by confidence
    precision = []
    recall = []
    TPs = []
    for pred in predicts:
        TPs.append(pred[1])
    TPs = np.array(TPs)
    for i in range(len(predicts)):
        accTP = np.sum(TPs[:i+1])
        accFP= i + 1 - accTP
        precision.append(accTP / (i + 1))
        recall.append(accTP / num_ground_truth)
    AP = 0
    for level in recall_levels:
        p_interp = 0
        for idx,r in enumerate(recall):
            if r > level:
                if precision[idx] > p_interp:
                    p_interp = precision[idx]
        AP += p_interp
    
    return (AP/ 11,total_time)

def evaluate_all_class(cfg, weights):
    run_time_all = 0
    AP_all = 0
    n = 0  #number of images for each class

    # load the weights to a model
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    # get the output layer
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    for i in range(len(classes)):
        filenames = getFileNames(classes[i])
        n += len(filenames)
        AP, run_time = evalutaion(cfg, weights, net, ln, filenames, i)
        run_time_all += run_time
        AP_all += AP
    return (AP_all / len(classes), run_time_all / n)


def write_ouput(output_file, mAP, runt_time, mAP_tiny, runt_time_tiny):
    with open(output_file,"w") as f:
        f.write('YOLO + \n')
        f.write( 'mAP: ' + str(mAP) + '\n')
        f.write( 'Run time: ' + str(runt_time) + '\n')
        f.write('\nTiny YOLO + \n')
        f.write( 'mAP: ' + str(mAP_tiny) + '\n')
        f.write( 'Run time: ' + str(runt_time_tiny) + '\n')

if __name__ == '__main__': 
    # evaluate yolo
    cfg = 'yolov2-voc.cfg'
    weights = 'yolov2-voc.weights'
    mAP, runt_time = evaluate_all_class(cfg, weights)

    # evaluate tiny
    cfg = 'yolov2-voc.cfg'
    weights = 'yolov2-voc.weights'
    mAP_tiny, runt_time_tiny = evalutaion(cfg, weights)

    output_file = 'result.txt'
    write_ouput(output_file, mAP, runt_time, mAP_tiny, runt_time_tiny)





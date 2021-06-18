from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
# For resizing our imaeges for yolov3 model
from yolov3_tf2.dataset import transform_images
# Helps us to convert boxes into the dep sort format
from yolov3_tf2.utils import convert_boxes

# Used for nom maximim supressions
from deep_sort import preprocessing
# For setting up deep association metrics
from deep_sort import nn_matching
# Hepls us in detecting object
from deep_sort.detection import Detection
# For writing the track information
from deep_sort.tracker import Tracker
# Import feature generation encoder
from tools import generate_detections as gdet

# import class names
class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

# For identifying object is same or not
# if the distance is larger than this then features are same for object in current frame and previous frame
max_cosine_distance = 0.5
# create feature library and store feature vectors.
nn_budget = None
# avoids to many object detection for same objects
nms_max_overlap = 0.8

# pre-trained CNN model for tracking object
model_filename = 'model_data/mars-small128.pb'
# create feature encoder for feature generation and pass a model who will generate feature of the boject
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# create deep association metric
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
# pass this metric into the tracker
tracker = Tracker(metric)

vid = cv2.VideoCapture('./data/video/video.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
# get fps from original video
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
# Store historical points inside queue
pts = [deque(maxlen=30) for _ in range(1000)]

counter = []
line_cross_set = set()
while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break

    height, width, _ = img.shape
    
    cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)

    # color format in oppencv is BGR and in tf, yolo use RBG format
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Height, Width, Channel and batch size (4d array)
    img_in = tf.expand_dims(img_in, 0)
    # default size for yolov3 is 416 
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)
    # boxes = 3D (1, 100, 4(x_center,y_center,height,width)
    # score = 2D
    # classes = 2D
    # nums = 1D total detected objects

    classes = classes[0] #get first raw only
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)

    # remove the packed zeros for boxes and scale it back to orginal size of the image.
    # Also covert array to list
    converted_boxes = convert_boxes(img, boxes[0])
    # generate feature vectors for detected object
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    
    # remove redundancy  
    detections = [detections[i] for i in indices]

    # propagate the track distribution one time step forward based on kalman filtering     
    tracker.predict()
    # updates kalman filter's parameters and feature set
    # also assess the target disappearance and new target appearance
    tracker.update(detections)

    # visualize the result
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
    
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        # Store center point according to track_id 
        pts[track.track_id].append(center)

        
        
        direction = ""
        # Draw motion path
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            # dx = int(bbox[2]) - int(bbox[0]) # for left and right direction
            dy = pts[track.track_id][j][1] - pts[track.track_id][j-1][1] #for up and down direction 
            direction = "right direction" if dy>0 else "wrong direction"
            
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        cv2.putText(img, class_name+"-"+str(track.track_id)+":"+direction, (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        
        center_y = int(((bbox[1])+(bbox[3]))/2)

        if center_y >= int(3*height/6+height/20):
            counter.append(int(track.track_id))
            if direction == "right direction":
                line_cross_set.add(int(track.track_id))

    total_count = len(set(counter))
    line_cross_count = len(line_cross_set)
    cv2.putText(img, "Vehicles Passed through line: " + str(line_cross_count), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(img, "Total Vehicle Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    # cv2.resizeWindow('output', 1024, 768)
    # cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
current_count = int(0)
vid.release()
out.release()
cv2.destroyAllWindows()
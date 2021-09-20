import os
import math
from re import S
import datetime as dt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import json
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/Part1.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

customConfig =  { "columns": 4 ,  "rows" : 2 , "BBox" : [], "segment": {}}
currentSegments = []
activeSegmets = {}
closedSegments = []
OCRtime = "Not started"
personList = []

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    #creating segment logger
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cellWidth = width/customConfig["columns"]
    cellHeight = height/customConfig["rows"]
    for y in range(0, customConfig['rows']):
        for x in range(0, customConfig['columns']):
            customConfig["BBox"].append({
                            "xmin": x*cellWidth, 
                            "ymin": y*cellHeight, 
                            "xmax": (x+1)*cellWidth, 
                            "ymax": (y+1)*cellHeight,
                            "center":[x*cellWidth + cellWidth/2, y*cellHeight + cellHeight/2],
                            "segment" : str(y)+str(x)
                        })
            customConfig["segment"][str(y)+str(x)] = {
                            "xmin": x*cellWidth, 
                            "ymin": y*cellHeight, 
                            "xmax": (x+1)*cellWidth, 
                            "ymax": (y+1)*cellHeight,
                            "log": {},
            }
    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))


    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()      
        
        if return_value:
    
            #frame for deep sort
            roi = frame[int(height*0.91):height-40, int(width*0.55):width-1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        # print('Frame #: ', frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        currentSegments.clear()
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
        
            #add to personList
            try:
                pos = personList.index(int(track.track_id))+1
            except:
                pos = len(personList)+1
                personList.append(int(track.track_id))
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(pos)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(pos),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
        
        # adding entries to segment logger
            distance = []

            for segment in customConfig["BBox"]:
                center1 = [(int(bbox[0])+int(bbox[2]))/2 , (int(bbox[1])+int(bbox[3]))/2]
                center2 = segment["center"]
                distanceX = abs(center1[0] - center2[0])
                distanceY = abs(center1[1] - center2[1])
                diagonal = math.sqrt(pow(distanceX,2) + pow(distanceY,2))
                distance.append([diagonal ,segment["segment"]])

               
            segmentID = min(distance)[1]
            box = customConfig["segment"][segmentID]
            cv2.rectangle(frame, (int(box["xmin"]), int(box["ymin"])), (int(box["xmax"]), int(box["ymax"])), (0,255,0) , 2)
            cv2.putText(frame,'A'+str(int(segmentID[0]) * customConfig["columns"] + int(segmentID[1]) + 1),
                (int(box["xmax"]-100), int(box["ymin"]+70)),0, 2, (0,255,0), 5)
            #Get Time
            roi = cv2.resize(roi, None, fx=0.5, fy=0.5)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = cv2.bitwise_not(roi)
            ret, roi = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY)

            config = "--psm 7"
            OCRtime = pytesseract.image_to_string(roi, config = config)
            personID = str(pos)
        #   add to logger
            currentSegments.append({"id":personID, "segment":segmentID, "time" : OCRtime})

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(pos), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        #############################################################################
        #get id of currentsegmet as a list
        currentList = []
        for segment in currentSegments:
            currentList.append(segment["id"])
        
        #check activeSegments
        activeList = activeSegmets.keys()
        newIDs = list(set(currentList).difference(activeList))
        closedIDs = []
        # closedIDs = list(set(activeList).difference(currentList))
        existingIDs = list(set(currentList).intersection(activeList))
        

        #add new entries to activeSegments
        for segment in currentSegments:
            if (segment["id"] in newIDs) :
                activeSegmets[segment["id"]]= {"segment":segment["segment"], "time":segment["time"]}
            elif ((segment["id"] in existingIDs) and (segment["segment"] != activeSegmets[segment["id"]]["segment"])):
                closedIDs.append(segment["id"])

        #remove closed ids
        for id in closedIDs:
            currentState = activeSegmets[id]
            entryTime = currentState["time"][ : -2]
            exitTime  = OCRtime[ : -2]            
            closedSegments.append({
                "person":id,
                "section":currentState["segment"],
                "EntryTime":entryTime,
                "ExitTime":exitTime,
                "duration": getDuration(entryTime, exitTime)
                });
            activeSegmets.pop(id)
        



        #########################################################################
        if not FLAGS.dont_show:
            reImg = cv2.resize(result, ( 960, 540 ))
            cv2.imshow("Output Video", reImg)
        

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    ###############-end-of-video-#########################
    for segment in activeSegmets.keys():
        closedSegments.append({"person":segment, "section":activeSegmets[segment]["segment"], "EntryTime":activeSegmets[segment]["time"][ : -2], "ExitTime":"N/A", "duration":"N/A"})
        
    closedSegments2 = list(map(formatLogger, closedSegments))

    with open('logs.json', 'w') as fp:
        json.dump(closedSegments2 , fp,  indent=4)
    cv2.destroyAllWindows()

personList = []

def getTime(tt):
    [_date , _time , _type] = tt.split(" ")
    [month, day, year] = _date.split("/")
    [hour, minute, second] = _time.split(":")
    if _type == 'PM':
        hour = int(hour) + 12
    return [int(year), int(month), int(day), int(hour), int(minute), int(second)]

def getDuration(start, end):
    st = getTime(start)
    et = getTime(end)
    a = dt.datetime(st[0], st[1], st[2], st[3], st[4], st[5])
    b = dt.datetime(et[0], et[1], et[2], et[3], et[4], et[5])
    return (b-a).total_seconds()


def formatLogger(item):
    section = item["section"]  
    sectionName = 'A'+str(int(section[0]) * customConfig["columns"] + int(section[1]) + 1)
    return {"person":"P"+item["person"], "section":sectionName, "EntryTime":item["EntryTime"], "ExitTime":item["ExitTime"], "duration":item["duration"]}



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

import os
import math
from re import S
import datetime as dt
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from threading import *
import webbrowser
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import json
import csv

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


customConfig =  { "columns": 4 ,  "rows" : 2 , "BBox" : [], "segment": {}}
currentSegments = []
activeSegmets = {}
closedSegments = []
OCRtime = "Not started"
personList = []
root = tk.Tk()
GUI_filePath = tk.StringVar()
GUI_columns = tk.StringVar()
GUI_rows = tk.StringVar()
inp_size = 416
inp_iou = 0.45
inp_score = 0.50
inp_output_format = 'XVID'
inp_model = 'yolov4'
inp_output = './output.avi'
closedSegments2 = []


def main(vid):
    out = None
    nms_max_overlap = 1.0
    input_size = 416
     # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    print("started")
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

    print("loading yolo")
    #load YOLO model
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']
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
    # # get video ready to save locally if flag is set
    if inp_output:
        # by default VideoCapture returns float instead of int
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*inp_output_format)
        out = cv2.VideoWriter(inp_output, codec, fps, (width, height))


    frame_num = 0
    # while video is running
    statusMessage.set("Video Processing, In progess")
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
            iou_threshold=inp_iou,
            score_threshold=inp_score
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
            NewRoi = cv2.resize(roi, None, fx=0.5, fy=0.5)
            NewRoi = cv2.cvtColor(NewRoi, cv2.COLOR_BGR2GRAY)
            NewRoi = cv2.bitwise_not(NewRoi)
            ret, NewRoi = cv2.threshold(NewRoi, 70, 255, cv2.THRESH_BINARY)

            config = "--psm 7"
            OCRtime = pytesseract.image_to_string(NewRoi, config = config)
            personID = str(pos)

            if GUI_showOCR.get() == 1:
                cv2.imshow("OCR of Time", NewRoi)

        #   add to logger
            currentSegments.append({"id":personID, "segment":segmentID, "time" : OCRtime})

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
            try:
                dur = getDuration(entryTime, exitTime) 
            except:
                dur = "Invalid"   
            closedSegments.append({
                "person":id,
                "section":currentState["segment"],
                "EntryTime":entryTime,
                "ExitTime":exitTime,
                "duration": dur
                });
            activeSegmets.pop(id)
        



        #########################################################################
        reImg = cv2.resize(result, ( 960, 540 ))
        cv2.imshow("Output Video", reImg)
        

        # if output flag is set, save video file
        if inp_output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        if cv2.getWindowProperty("Output Video", 4) < 1:
            break

    ###############-end-of-video-#########################
    exitTime  = OCRtime[ : -2]  
    for segment in activeSegmets.keys():
        try:
            dur = getDuration(activeSegmets[segment]["time"][ : -2], exitTime)
        except:
            dur = "Invalid"
        closedSegments.append({"person":segment, "section":activeSegmets[segment]["segment"], "EntryTime":activeSegmets[segment]["time"][ : -2], "ExitTime":exitTime, "duration":dur})
    global closedSegments2     
    closedSegments2 = list(map(formatLogger, closedSegments))

    cv2.destroyAllWindows()


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

def browseVideo():
    fln = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select video file", filetypes = (("video files","*.mp4"),("all files","*.*")))
    GUI_filePath.set(fln)


def beginProcess():
    if (GUI_filePath.get() != ""):
        statusMessage.set("Loading ML models, Please Wait...")
        processButton['state'] = tk.DISABLED
        customConfig["columns"] = int(GUI_columns.get())
        customConfig["rows"] = int(GUI_rows.get())
        global currentSegments
        global activeSegmets
        global closedSegments
        global closedSegments2
        currentSegments = []
        activeSegmets = {}
        closedSegments = []
        closedSegments2 = []
        vid = cv2.VideoCapture(GUI_filePath.get())
        global inp_output
        inp_output =  None if GUI_exportVideo.get() == 0 else './output.avi'
        main(vid)
        statusMessage.set("Processing Complete, you can export now.")
        processButton['state'] = tk.NORMAL
    else:
        statusMessage.set("Please load video, before processing")

def beginJSONExport():
    with open('logs.json', 'w') as fp:
        json.dump(closedSegments2 , fp,  indent=4)

def beginCSVExport():
    keys = closedSegments2[0].keys()
    with open('logs.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(closedSegments2)
    

def beginThread():
    # Call work function
    t1=Thread(target=beginProcess)
    t1.daemon = True
    t1.start()

def callback(url):
    webbrowser.open_new_tab(url)

def createScreen():

    wrapper1 = tk.LabelFrame(root, text="Load Video")
    wrapper1.pack(fill="both", expand="yes", padx=2)
    
    tk.Entry(wrapper1, textvariable=GUI_filePath).place(x = 5 ,y = 10 , width = 250 , height = 30)
    tk.Button(wrapper1, text="Browse", command=browseVideo).place(x = 265,y = 10, width = 70) 


    wrapper2 = tk.LabelFrame(root, text="Process")
    wrapper2.pack(fill="both", expand="yes", padx=2, pady=2) 
    GUI_columns.set("4")
    GUI_rows.set("2")
    tk.Label(wrapper2, text = "Columns :").place(x = 5 ,y = 15)  
    tk.Entry(wrapper2, textvariable=GUI_columns).place(x = 70 ,y = 15 , width = 70 , height = 20)
    tk.Label(wrapper2, text = "rows :").place(x = 145,y = 15)  
    tk.Entry(wrapper2, textvariable=GUI_rows).place(x = 185 ,y = 15 , width = 70 , height = 20)
    global statusMessage 
    statusMessage = tk.StringVar()
    tk.Label(wrapper2, textvariable = statusMessage, fg="green").place(x = 5 ,y = 40)
    global processButton
    processButton = tk.Button(wrapper2, text="Process", command=beginThread)
    processButton.place(x = 265,y = 10, width = 70, height=30 )

    wrapper3 = tk.LabelFrame(root, text="Export")
    wrapper3.pack(fill="both", expand="yes", padx=2, pady=2)

    global GUI_exportVideo
    global GUI_showOCR
    GUI_exportVideo = tk.IntVar()
    GUI_exportVideo.set(1)

    GUI_showOCR = tk.IntVar()
    GUI_showOCR.set(0)
    tk.Checkbutton(wrapper3, text="Show OCR", variable=GUI_showOCR).place(x = 5,y = 5)
    tk.Checkbutton(wrapper3, text="Video Export", variable=GUI_exportVideo).place(x = 5,y = 30)
    tk.Button(wrapper3, text="Export JSON", command=beginJSONExport).place(x = 130,y = 5)
    tk.Button(wrapper3, text="Export CSV", command=beginCSVExport).place(x = 230,y = 5)

    link = tk.Label(wrapper3, text="Github-SourceCode",font=('Helveticabold', 10), fg="blue", cursor="hand2")
    link.place(x = 160 ,y = 40)
    link.bind("<Button-1>", lambda e: callback("https://github.com/solid-droid/Customer-Interaction-in-shop"))

    

if __name__ == '__main__':
    try:
        createScreen() 
        root.title("Customer Interaction Tracker")
        root.geometry("350x250")
        root.mainloop()
    except SystemExit:
        pass

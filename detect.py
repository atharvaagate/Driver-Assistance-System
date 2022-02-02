from __future__ import division 
import time
import torch
import numpy as np
import pickle as pkl
import random
import pandas as pd
import cv2
from torch.autograd import Variable
import os
from util import *
from darknet import Darknet
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt



'''
def arg_parse() :
    parser = argparse.ArgumentParser(description= "Yolo Object Detector")
    parser.add_argument("--bs" ,dest= "bs",help= "Batch Size",default= 1)
    parser.add_argument("--conf" ,dest= "conf",help= "Confidense criteria fro Object Detection",default= 0.5)
    parser.add_argument("--nms_thresh" ,dest= "nms_thresh",help= "NonMax Supression Threshold",default= 0.4)
    parser.add_argument("--cfg" ,dest= "cfg_file",help= "Config File",default= "cfg/yolov3.cfg", type= str)
    parser.add_argument("--weights" ,dest= "weights",help= "Model Weights",default= "weights/yolov3.weights", type= str)
    parser.add_argument("--res" ,dest= "res",help= "Resolution of image",default= "416", type= str)
    parser.add_argument("--video" ,dest= "video",help= "Video File",default= "video.mp4", type= str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
nms_thresh = int(args.nms_thresh)
conf = int(args.conf)
weights = args.weights
cfg = args.cfg_file
resolution = int(args.res)
start = 0

CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

print("Loading the Network...")

model = Darknet(cfg)
model.load_weights(weights)

print("Model Loaded Completely!")

model.net_info['height'] = resolution

inp_dim = int(model.net_info['height'])

assert inp_dim > 32
assert inp_dim %32 == 0

if CUDA :
    model.cuda()
    print("Model in cuda")

model.eval()



def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    #label = "{0}".format(classes[cls])
    label = "p"
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img






















video = args.video

cap = cv2.VideoCapture(0)


frames = 0

while cap.isOpened() :
    ret, frame = cap.read()

    if ret :
        img = prep_image(frame, inp_dim)
        img_dim = frame.shape[1], frame.shape[0]
        img_dim = torch.FloatTensor(img_dim).repeat(1,2)

        if CUDA :
            img = img.cuda()
            img_dim = img_dim.cuda()

        with torch.no_grad() :
            output = model(img, CUDA)

        output = write_results(output, conf, num_classes, nms_thresh)

        if type(output) == int :
            frames+=1
            print(f"FPS of the video is { frames / (time.time() - start)}")

            cv2.imshow("frame", frame)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = img_dim.repeat(output.size(0), 1)
        scale_factor = torch.min(416/im_dim, 1)[0].view(-1,1)

        output[:,[1,3]] -= (inp_dim - scale_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scale_factor*im_dim[:,1].view(-1,1))/2

        output[:,1:5] /= scale_factor


        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

        classes = load_classes("data/coco.names")
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x : write(x, frame), output))

        cv2.imshow("frame", frame)
        #cv2.imwrite('det.jpg', frame)
        #plt.imshow(frame)
        #time.sleep(5)
        #break

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

        frames+=1

        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

    else :
        break

'''





def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "weights/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--video", dest = "videofile", help = "Video file to     run detection on", default = "video.mp4", type = str)
    
    return parser.parse_args()
    
args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()



num_classes = 80
classes = load_classes("data/coco.names")



#Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode
model.eval()



def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cl = int(x[-1])
    print(cl)
    color = random.choice(colors)
    try :
        label = "{0}".format(classes[cl])
    except :
        label = "Unidentified"
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


#Detection phase

videofile = args.videofile #or path to the video file. 

cap = cv2.VideoCapture("video.mp4")  
cap.set(cv2.CAP_PROP_FPS, 100)
fps = int(cap.get(100))

#cap = cv2.VideoCapture(0)  #for webcam

assert cap.isOpened(), 'Cannot capture source'

frames = 0  
start = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
        #cv2.imshow("a", frame)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()
        
        with torch.no_grad():
            output = model(Variable(img, volatile = True), CUDA)
        output = write_results(output, confidence, num_classes, nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        
        
        

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    
        
        

        classes = load_classes('data/coco.names')
        colors = pkl.load(open("pallete", "rb"))

        list(map(lambda x: write(x, frame), output))
        cv2.imwrite("det.jpg", frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     


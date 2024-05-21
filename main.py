import sys
from pathlib import Path
import os
import torch
import cv2
import pyttsx3
#import RPi.GPIO as GPIO
import time
#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)
# GPIO.setup(12, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(16, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(20, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# GPIO.setup(10, GPIO.IN)# mode button
# GPIO.setup(11, GPIO.IN)# confirm button

#GPIO.setup(23, GPIO.OUT)
#GPIO.output(23, True)
engine = pyttsx3.init()
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression


class Currency:
    def __init__(self):
        self.conf_thres = 0.4
        self.iou_thres = 0.45
        self.imsize = 416
        self.max_bag =1
        self.model = attempt_load("/home/rock/Desktop/Hearsight/English/Currency/yolov5/currency3.pt")#, map_location='cpu')
        self.stride = int(self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def inference(self):
        try:
            a="ffmpeg -f v4l2 -video_size 1280x720 -i /dev/video1 -frames 1 /home/rock/Desktop/Hearsight/English/Currency/yolov5/1.jpg"
            b=os.system(a)
            imgsz = check_img_size(self.imsize, s=self.stride)
            k = 0
            dataset = LoadStreams('/home/rock/Desktop/Hearsight/English/Currency/yolov5/1.jpg', img_size=imgsz, stride=self.stride, auto=True)
            all_preds = []
            for path, img in dataset:
                #cv2.imshow("im0s",im0s[0])
                cv2.waitKey(10)
                img = torch.from_numpy(img).to('cpu')
                img = img.float()
                img = img / 255.0
                if len(img.shape) == 3:
                    img = img[None]
                pred = self.model(img, augment=False, visualize=False)[0]
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=8)
                for i, det in enumerate(pred):
                    preds = []
                    for c in det[:, -1].unique():
                        preds.append(int(self.names[int(c)][1:]))
                    k += 1
                    all_preds.append(preds)
                    if k == self.max_bag:
                        b = {10: "Ten", 20: "Twenty", 50: "Fifty", 100: "Hundered", 200: "Twohundered", 500: "Fivehundered", 2000: "Twothousand"}
                        finalized = max(all_preds,key=all_preds.count)
                        print(finalized)
                        obj = "image_captured"
                        engine.say(obj)
                        engine.runAndWait()
#                         a1="omxplayer  --vol 600 -o local Module10_audio/image_captured.wav"
#                         d=os.system(a1)
                        GPIO.output(23, False)
                        for a in finalized: 
                            print(b[a])
                            go = (b[a])
                            engine.say(go)
                            engine.runAndWait()
#                             mytext= "espeak " +b[a]+"RupeesNote"" -g 10 -s 180 -v en-in -w Module10_audio/Currency.wav"
#                             os.system(mytext)
#                             a1="omxplayer  --vol 600 -o local Module10_audio/Currency.wav"
#                             os.system(a1)
                        print(str(sum(finalized)))
                        ko = (str(sum(finalized)))
                        engine.say(ko)
                        engine.runAndWait()
#                         mytext= "espeak " +"Total"+(str(sum(finalized)))+" -g 10 -s 180 -v en-in -w Module10_audio/Sum.wav"
#                         os.system(mytext)
#                         a1="omxplayer  --vol 600 -o local Module10_audio/Sum.wav"
#                         os.system(a1)
                        return
        except:
            print("tq")
#             a1="omxplayer  --vol 600 -o local Module10_audio/camera_not_working.wav"
#             d=os.system(a1)
    

if __name__ == "__main__":
    c = Currency()
    c.inference()
    
# while True:
#        a1="omxplayer  --vol 600 -o local Module10_audio/image_captured.wav"
#        os.system(a1)
#        if(GPIO.input(16)==False):
#            c.__init__()
#            
#        if(GPIO.input(20)==False):
#             
#             break
#        time.sleep(3)


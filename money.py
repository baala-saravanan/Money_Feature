import torch
import re
import os
import time
from PIL import Image
from pydub import AudioSegment
import sys
import subprocess
import cv2
import pyttsx3
from play_audio import GTTSA

play_audio = GTTSA()
engine = pyttsx3.init()

# Set the directory for YOLOv5 models
torch.hub.set_dir('/home/rock/Desktop/Hearsight/English/money/yolov5/')

# Load the YOLOv5 model
#model = torch.hub.load('/home/rock/Desktop/Hearsight/English/money/yolov5/',
#                       'custom',
#                       path="/home/rock/Desktop/Hearsight/English/money/yolov5/currency3.pt",
#                       force_reload=True,
#                       source='local')
#/home/rock/Desktop/Hearsight/English/money/yolov5/best.pt
#/home/rock/Desktop/Hearsight/best.pt

model_1 = torch.hub.load('/home/rock/Desktop/Hearsight/English/money/yolov5/',
                       'custom',
                       path="/home/rock/Desktop/Hearsight/mycroft-precise/test/scripts/best.pt",
                       force_reload=True,
                       source='local')
#model.conf = 0.6
#model.iou = 0.5
#model_1.conf = 0.6
#model_1.iou = 0.5
model_1.conf = 0.65
model_1.iou = 0.45

img_path = '/home/rock/Desktop/Hearsight/English/money/yolov5/1.jpg'

class Money:
#    def predict_objects(self, model, img_path):
#        img = Image.open(img_path)
#        out = model(img)
#        results = out.pandas().xyxy[0].sort_values('xmax').name.values
#        return results
    
    def predict_objects_1(self, model_1, img_path):
        img_1 = Image.open(img_path)
        out_1 = model_1(img_1)
        results_1 = out_1.pandas().xyxy[0].sort_values('xmax').name.values
        return results_1
        
#    def say_number(self, number):
#        play_audio.play_machine_audio(f'number_{number}.mp3')
        
    def say_number_1(self, number_1):
        play_audio.play_machine_audio(f'number_{number_1}.mp3')

#    def pred(self):
#        cap = cv2.VideoCapture(1)  # Change the camera index as needed
#        if not cap.isOpened():
#            play_audio.play_machine_audio("camera_is_not_working_so_switch_off_the_HearSight_device_for_some_time_and_then_start_it_again.mp3")
#            return
#        cap.release()
#        cv2.destroyAllWindows()
#        
#        play_audio.play_machine_audio('hold your notes steady.mp3')
#        
#        if os.path.exists(img_path):
#            os.remove(img_path)
#
#        # Capture the image using OpenCV (cv2)
#        cap = cv2.VideoCapture(1)
#        ret, frame = cap.read()
#        if not ret:
#            play_audio.play_machine_audio("image_capture_failed_so_retake_it_again.mp3")
#            return
#        cv2.imwrite(img_path, frame)
#        cap.release()
#        cv2.destroyAllWindows()
#
#        play_audio.play_machine_audio('Image_captured.mp3')
#        play_audio.play_machine_audio('Processing.mp3')
#
#        results = self.predict_objects(model, img_path)
#        numbers = re.findall(r'\d+', str(results))
#        print(numbers)
#        # Check if the 'numbers' list is empty
#        if not numbers:
#            play_audio.play_machine_audio('Unclear Image.mp3')
#        else:
#            numbers = numbers[0]  # Extract the first number
#            print(numbers)
#            number = int(numbers)
#            if number >= 2000:
#                play_audio.play_machine_audio('Unclear Image.mp3')
#            elif number >= 1000:
#                thousands = number // 1000
#                self.say_number(thousands)
#                play_audio.play_machine_audio('number_1000.mp3')
#                number -= thousands * 1000
#            elif number >= 100:
#                hundreds = number // 100
#                self.say_number(hundreds)
#                play_audio.play_machine_audio('number_100.mp3')
#                number -= hundreds * 100
#            elif number >= 10:
#                tens = number // 10
#                self.say_number(tens * 10)
#                number -= tens * 10
#
#        os.remove(img_path)
    
    def pred_1(self):
        cap_1 = cv2.VideoCapture(1)  # Change the camera index as needed
        if not cap_1.isOpened():
#            play_audio.play_machine_audio("camera_is_not_working_so_switch_off_the_HearSight_device_for_some_time_and_then_start_it_again.mp3")
#            play_audio.play_machine_audio("check_your_connection_and_proceed.mp3")
            play_audio.play_machine_audio("hold_on_connection_in_progress_initiating_shortly.mp3")
            play_audio.play_machine_audio("Thank You.mp3")
            subprocess.run(["reboot"])
            return
        cap_1.release()
        cv2.destroyAllWindows()
        
#        play_audio.play_machine_audio('hold your notes steady.mp3')
        
        if os.path.exists(img_path):
            os.remove(img_path)

        # Capture the image using OpenCV (cv2)
        cap_1 = cv2.VideoCapture(1)
        ret_1, frame_1 = cap_1.read()
        if not ret_1:
            play_audio.play_machine_audio("image_capture_failed_so_retake_it_again.mp3")
            return
        cv2.imwrite(img_path, frame_1)
        cap_1.release()
        cv2.destroyAllWindows()

#        play_audio.play_machine_audio('Image_captured.mp3')
#        play_audio.play_machine_audio('Processing.mp3')

        results_1 = self.predict_objects_1(model_1, img_path)
        numbers_1 = re.findall(r'\d+', str(results_1))
        print(numbers_1)
        
        # Replace '2000' with '0' in the list
        numbers_1 = ['0' if number == '2000' else number for number in numbers_1]
        print(numbers_1)

        total_1 = sum(map(int, numbers_1))
        print(total_1)
        
        if total_1 == 0:
            play_audio.play_machine_audio('Unclear Image.mp3')
        else:
#            play_audio.play_machine_audio('Image_captured.mp3')
#            play_audio.play_machine_audio('Processing.mp3')
                    
            for number_1 in numbers_1:
                number_1 = int(number_1)
                print(number_1)
                if number_1 >= 1000:
                    thousands_1 = number_1 // 1000
                    self.say_number_1(thousands_1)
                    play_audio.play_machine_audio('number_1000.mp3')
                    number_1 -= thousands_1 * 1000
                elif number_1 >= 100:
                    hundreds_1 = number_1 // 100
                    self.say_number_1(hundreds_1)
                    play_audio.play_machine_audio('number_100.mp3')
                    number_1 -= hundreds_1 * 100
                elif number_1 >= 10:
                    tens_1 = number_1 // 10
                    self.say_number_1(tens_1 * 10)
                    number_1 -= tens_1 * 10
                elif number_1 > 0:  # Adjusted this condition
                    self.say_number_1(number_1)
                
#        if total_1 == 0:
#            play_audio.play_machine_audio('Unclear Image.mp3')
        
        while total_1 > 0:
            play_audio.play_machine_audio('total.mp3')
#            engine.setProperty('voice', 'english_rp+f3')
#            engine.setProperty('rate', 200)
            engine.setProperty('voice', 'en-gb')
            engine.setProperty('rate', 140)
            engine.say(total_1)
            engine.runAndWait()
            break

        os.remove(img_path)
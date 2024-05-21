import os

import wget
import numpy as np
import cv2

from PIL import Image
import pickle
from config import *

import tensorflow as tf


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

import time
import shutil

import sys
import face_recognition

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class Recognizer:
    def __init__(self):
        # create base storage dir if don't exist
        if not os.path.isdir(BASE_DIR):
            os.makedirs(BASE_DIR)

        if not os.path.isdir(RAW_IMG_DIR):
            os.makedirs(RAW_IMG_DIR)

        if not os.path.isdir(EMBEDDINGS_DIR):
            os.makedirs(EMBEDDINGS_DIR)

        self.persons = os.listdir(RAW_IMG_DIR)
        self.detector = face_recognition.face_locations
        self.feature_extractor = face_recognition.face_encodings
        self.normalizer = Normalizer(norm='l2')
        self.label_enc = LabelEncoder()
        self.model = SVC(kernel='linear', probability=True)
        self.VERBOSE = VERBOSE

        if os.path.isfile(RECOGNIZER_PATH):
            self.recognizer = pickle.load(open(RECOGNIZER_PATH, 'rb'))
            with open(MAPPING_PATH, "rb") as f:
                unpickler = pickle.Unpickler(f)
                self.label_dict = unpickler.load()

    def augument(self, pixels):
        images = [pixels]
        seed = (1, 2)
        # Random brightness
        image = tf.image.stateless_random_brightness(
            pixels, max_delta=0.4, seed=seed)
        images.append(image.numpy())
        image = tf.image.stateless_random_contrast(
            pixels, lower=0.3, upper=0.8, seed=seed)
        images.append(image.numpy())

        image = tf.image.stateless_random_jpeg_quality(pixels, min_jpeg_quality=40, max_jpeg_quality=60, seed=seed)
        images.append(image.numpy())

        image = tf.image.stateless_random_saturation(pixels, lower=0.4, upper=0.6, seed=seed)
        images.append(image.numpy())

        return images

    def collect_images(self, path, name):
        camera = cv2.VideoCapture(CAM_PORT)
        i = 1
        while i <= NO_IMG_PER_PERSON:
            check, frame = camera.read()
            cv2.imshow("press c to capture", frame)
            key = cv2.waitKey(1)
            if key == ord('c'):
                if self.check_face(frame):
                    cv2.imwrite(filename=f"{path}/{name}{i}.jpg", img=frame)
                    print(f"Image {i} captured successfully")
                    i += 1
                else:
                    print("Unclear image, please retake with proper lighting and with only 1 person")
        cv2.destroyAllWindows()

    def check_face(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.detector(rgb, model='hog')
        if len(result) == 1:
            return True
        else:
            return False

    def process_raw_imgs(self, path, name):
        embeddings = []
        for img in os.listdir(path):
            image = Image.open(os.path.join(path,img))
            image = image.convert('RGB')
            pixels = np.asarray(image)
            result = self.detector(pixels, model='hog')
            encoding = self.feature_extractor(pixels, result)
            embeddings.append(encoding[0])
            aug_imgs = self.augument(pixels)

            for aug_img in aug_imgs:
                encoding = self.feature_extractor(aug_img, result)
                embeddings.append(encoding[0])
        embeddings = np.array(embeddings)
        embeddings = self.normalizer.transform(embeddings)
        print(embeddings.shape)
        np.save(f"{EMBEDDINGS_DIR}/{name}.npy", embeddings)

    def build_model(self):
        embeddings = []
        y = []
        i = 0
        label_dict = {}
        for emb in os.listdir(EMBEDDINGS_DIR):
            embeddings += list(np.load(os.path.join(EMBEDDINGS_DIR, emb)))
            cls = emb.split('.')[0]
            y += [i] * NO_IMG_PER_PERSON * 6
            label_dict[i] = cls
            i += 1
        embeddings = np.array(embeddings)
        self.label_enc.fit(y)
        y = self.label_enc.transform(y)
        self.model.fit(embeddings, y)
        with open(MAPPING_PATH, 'wb') as fp:
            pickle.dump(label_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.model, open(RECOGNIZER_PATH, 'wb'))
        self.recognizer = self.model
        self.label_dict = label_dict
        self.persons = os.listdir(RAW_IMG_DIR)

    def add_person(self):
        name = input("Enter person name: ").strip()
        while name in self.persons:
            name = input("Person name already exists, try new name: ").strip()
        path = os.path.join(RAW_IMG_DIR, name)
        os.makedirs(path)
        self.collect_images(path, name)
        start_time = time.time()
        self.process_raw_imgs(path, name)
        self.persons = os.listdir(RAW_IMG_DIR)
        if len(self.persons) > 1:
            self.build_model()
        print(f"Successfully added person {name}")
        if self.VERBOSE:
            print(f"Time taken for adding a person took {time.time() - start_time} seconds")

    def get_threshold(self):
        thres = THRESHOLD_BASE
        for i in range(5, len(self.persons), 5):
            thres -= 1
        return thres/100

    def recognize(self):
        camera = cv2.VideoCapture(CAM_PORT)
        THRESHOLD = self.get_threshold()
        while True:
            check, frame = camera.read()
            cv2.imshow("detector (press r to recognise)", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                start_time = time.time()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.detector(rgb, model='hog')
                print("No of faces detected:", len(results))
                print(f"Time taken to recognize: {time.time()-start_time} seconds")
                rec = 0
                for result in results:
                    new_embed = self.feature_extractor(rgb, [result])
                    x1, x2, y1, y2 = result
                    new_embed = self.normalizer.transform(new_embed)
                    y_pred = self.recognizer.predict_proba(new_embed)[0]
                    confidence = np.max(y_pred)
                    if confidence >= THRESHOLD:
                        person = self.label_dict[np.argmax(y_pred)]
                        print(person, int(confidence * 100))
                        rec += 1
                        cv2.putText(frame, f"{person}-{int(confidence * 100)}%", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=3)
                    print("Recognized person:", rec)
                cv2.destroyAllWindows()
                cv2.imshow("Prediction Window", frame)
                print(f"Recognition + detection time of all faces {time.time() - start_time}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def list_persons(self):
        print("#   \tName")
        for key, value in self.label_dict.items():
            print(key+1, '\t',value)
        print("*"*10,'\n')

    def remove_person(self):
        self.list_persons()
        inp = input("Type number or name of person to remove:").strip()
        if inp.isdigit():
            inp = int(inp) - 1
            name = self.label_dict.get(inp, None)
        else:
            name = inp
        if name in self.label_dict.values():
            confirm = input(f"Are you sure you want to delete {name}? [y/n]").strip()
            if confirm == 'y' or confirm == 'Y':
                os.remove(os.path.join(EMBEDDINGS_DIR, f"{name}.npy"))
                shutil.rmtree(os.path.join(RAW_IMG_DIR, name))
                self.persons = os.listdir(RAW_IMG_DIR)
                if len(self.persons) > 1:
                    self.build_model()
                print(f"{name} removed successfully!")
                return True
        else:
            print(f"{name} person not found, enter valid name or number")
            return False


recognizer = Recognizer()


def initaial_stager():
    while len(recognizer.persons) < 2:
        inp = input("Type 'a' to start adding people or 'q' to quit: ").strip().lower()
        if inp == 'a':
            recognizer.add_person()
        else:
            return

initaial_stager()
while len(recognizer.persons) >= 2:
    inp = input("Press 1 to add person\nPress 2 to start recognition\nPress 3 to list recognizable persons"
                "\nPress 4 to delete a person \nPress 5 to quit\nEnter Choice: ").strip()
    if inp == '1':
        recognizer.add_person()
    elif inp == '2':
        recognizer.recognize()
    elif inp == '3':
        recognizer.list_persons()
    elif inp == '4':
        check = recognizer.remove_person()
        while not check:
            again = input("Press 1 to try again\nPress 2 to move to main menu\nEnter choice:").strip()
            if again == "1":
                check = recognizer.remove_person()
            else:
                break
        if len(recognizer.persons) < 2:
            initaial_stager()
            if len(recognizer.persons) < 2:
                break
    else:
        break
import os
import cv2
import time
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from myPCA import *
from event import *

DATA_PATH = 'SSD_data'
IMG_PATH = 'images'
PROTOTXT = os.path.join(DATA_PATH, 'deploy.prototxt')
CAFFEMODEL = os.path.join(DATA_PATH, 'res10_300x300_ssd_iter_140000_fp16.caffemodel')
THRESHOLD = 0.8


class FaceDetector:
    def __init__(self, proto, model):
        self.network = cv2.dnn.readNet(proto, model)

    def query(self, image, threshold=THRESHOLD):
        target_size = 300, 300

        input_image = cv2.resize(image, target_size)
        image_blob = cv2.dnn.blobFromImage(input_image)

        self.network.setInput(image_blob)
        detections = self.network.forward()

        results = detections[0][0]
        results = list(filter(lambda x: x[2] > threshold, results))

        return results

    def draw(self, image, results):
        image = image.copy()

        height, width, _ = image.shape
        size = np.array([width, height, width, height])
            

        for result in results:
            conf = result[2]
            box = result[3:7] * size
            startX, startY, endX, endY = box.astype('int')

            cv2.putText(image, str(conf), (startX, startY - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        return image

    def get_images(self, image, results):
        images = []

        height, width, _ = image.shape
        size = np.array([width, height, width, height])

        for result in results:
            box = result[3:7] * size
            startX, startY, endX, endY = box.astype('int')

            images.append(image[startY:endY, startX:endX])

        return images


class FaceCaptureFrame(tk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.fps = 1
        self.camera = cv2.VideoCapture(0)
        self.faceDetector = FaceDetector(PROTOTXT, CAFFEMODEL)
        self.temp = False

        self.draw()

    def __del__(self):
        self.camera.release()

    def draw(self):
        # self.camera.set(3, 100)
        # self.camera.set(4, 100)

        self.label = tk.Label(self, text="Capture the face!")
        self.label.grid(column=0, row=0, columnspan=2)

        self.image = tk.Label(self)
        self.image.grid(column=0, row=1, columnspan=2)
        self.after(self.fps, self.update)

        tk.Button(self, text="Capture", command=self.capture).grid(column=0, row=2)

        self.grid()

    def get_image(self):
        ret, frame = self.camera.read()

        if not ret:
            return
        frame = cv2.resize(frame, (700, 500))
        frame = cv2.flip(frame, 1)
        return frame

    def update(self):
        frame = self.get_image()
        if self.temp:
            frame = cv2.convertScaleAbs(frame, 1, 20)

        frame = self.faceDetector.draw(frame, self.faceDetector.query(frame))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame))

        self.image.config(image=imgtk)
        self.image.image = imgtk

        self.after(self.fps, self.update)

    def capture(self):
        frame = self.get_image()
        results = self.faceDetector.get_images(frame, self.faceDetector.query(frame))

        for image in results:
            timestamp = int(time.time() * 1000000)
            filename = f'{timestamp}.jpg'
            messages = f'File saved ({filename})'

            self.label.config(text=messages)
            self.label.text = messages

            check(cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (120, 150)))
            # cv2.imwrite(filename, image)
            
        self.temp = True
        self.after(1000, lambda: setattr(self, 'temp', False))
        # self.pack_forget()
        # self.after(1000, self.pack)

def check(test_image):
    score = []
    for i in range(4):
        pca = pca_array[i]
        test_image = np.array(test_image, np.float32)
        image = test_image - average[i]
        image = image.reshape(18000, 1)
        image_value = transform[i].T @ image

        x = 0
        for j in range(count[i], count[i] + 10):
            arr = pca[:, x].reshape(index[i], 1)
            sum = 0

            for k in range(index[i]):
                sum += (image_value[k, 0] - arr[k, 0]) ** 2

            sum **= 1 / 2

            if j % 10 == 0 or min_array > sum:
                min_array = sum
                min_number = j
            x += 1
        print(f"{min_number} {min_array}")
        score.append(min_array)
    print(score)
    count = 0
    for i in range(4):
        if 1500 < score[0]: count += 1

    if count > 1: success()
    else: fail()

def gui():
    window = tk.Tk()
    window.title('Face Capture')
    window.geometry('800x600')

    faceCaptureFrame = FaceCaptureFrame(window)
    tk.Button(faceCaptureFrame, text="Quit", command=window.destroy).grid(column=1, row=2)
    faceCaptureFrame.pack()

    window.mainloop()


def main():
    faceDetector = FaceDetector(PROTOTXT, CAFFEMODEL)

    image = cv2.imread(input('[INPUT] Image src: '))
    image = faceDetector.draw(image, faceDetector.query(image))

    cv2.imshow('image', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    average = []
    difference = []
    index = []
    count = []
    transform = []
    pca_array = []

    for i in range(4):
        average.append(cv2.imread(f"./train_file/average{i}.jpg", cv2.IMREAD_GRAYSCALE))
        difference.append(cv2.imread(f"./train_file/difference{i}.jpg", cv2.IMREAD_GRAYSCALE))

        with open(f"./train_file/index{i}.txt", "r") as f:
            idx, cut = tuple(map(int, f.read().split(',')))
            index.append(idx)
            count.append(cut)

        transform.append(np.load(f"./train_file/transform{i}.npy"))
        pca_array.append(np.load(f"./train_file/pca_array{i}.npy"))

    
    print('average')
    print(average)
    print('difference')
    print(difference)
    print('index')
    print(index)
    print('count')
    print(count)
    print('transform')
    print(transform)
    print('pca_array')
    print(pca_array)

    gui()
    

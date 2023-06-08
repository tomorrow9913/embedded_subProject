import os
import cv2
import time
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

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
        self.camera.set(3, 100)
        self.camera.set(4, 100)

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

            cv2.imwrite(filename, image)

        self.temp = True
        self.after(1000, lambda: setattr(self, 'temp', False))
        # self.pack_forget()
        # self.after(1000, self.pack)

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
    gui()

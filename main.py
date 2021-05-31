import os
import cv2
import numpy as np
import faceRecognition as fr
from tkinter import *
from tkinter import messagebox as mb

def main():
        # This module captures images via webcam and performs face recognition
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('trainingData.yml')  # Load saved training data


        cap = cv2.VideoCapture(0)

        while True:
            ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
            faces_detected, gray_img = fr.faceDetection(test_img)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)

            resized_img = cv2.resize(test_img, (1280,720))
            cv2.imshow('FACE DETECTING', resized_img)
            cv2.waitKey(10)

            for face in faces_detected:
                (x, y, w, h) = face
                roi_gray = gray_img[y:y + w, x:x + h]
                label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
                fr.draw_rect(test_img, face)
                if confidence < 30:  # If confidence less than 37 then don't print predicted face text on screen
                    cap.release()
                    cv2.destroyAllWindows
                    return 0


def okno():
    window = Tk()
    window.title("Авторизация")
    window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))

    def clicked():
        main()
        mb.showinfo(' ', 'Лицо распознано')
        window.quit()

    text = Text(width=100, height=50)
    text.pack()
    text.insert(1.0, "Добро пожаловать!")

    text.tag_add('title', 1.0, '1.end')
    text.tag_config('title', justify=CENTER,
                    font=("Verdana", 24, 'bold'))
    btn = Button(window, text='Пройти авторизацию', command=clicked)

    btn.place(relx=0.5, rely=0.5, anchor=CENTER)

    label = Label(height=3)
    label.pack()

    window.mainloop()

okno()
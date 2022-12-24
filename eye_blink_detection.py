from imutils.video import VideoStream
import cv2
import time
import f_detector
import imutils
import numpy as np
from pynput.keyboard import Key, Controller
import csv
from datetime import datetime

detector = f_detector.eye_blink_detector()
COUNTER = 0
TOTAL = 0
CLOSED = 0
EYES_CLOSED_STATE = 0
BLINK_LOGGER_FILENAME = "blinks_" + \
    str(datetime.now()).replace(' ', '_').replace(":", "-")+".csv"
BLINKLIST = []
MA_TIME = 30  # 30 seconds


# ----------------------------- video -----------------------------
vs = VideoStream(src=0).start()
keyboard = Controller()
with open(BLINK_LOGGER_FILENAME, 'w', newline='') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['BLINK_ID', 'TIMESTAMP', 'EVENT'])
    while True:
        start_time = time.time()
        im = vs.read()
        im = cv2.flip(im, 1)
        im = imutils.resize(im, width=720)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # detect face
        rectangles = detector.detector_faces(gray, 0)
        boxes_face = f_detector.convert_rectangles2array(rectangles, im)
        if len(boxes_face) != 0:
            # box face
            areas = f_detector.get_areas(boxes_face)
            index = np.argmax(areas)
            rectangles = rectangles[index]
            boxes_face = np.expand_dims(boxes_face[index], axis=0)
            # detect blinks
            totalBefore = TOTAL
            COUNTER, TOTAL, CLOSED, BLINK = detector.eye_blink(
                gray, rectangles, COUNTER, TOTAL)

            for i in range(len(BLINKLIST)-1):
                if (datetime.now() - BLINKLIST[i]).seconds > MA_TIME:
                    BLINKLIST.pop(i)
            if TOTAL > totalBefore:
                BLINKLIST.append(datetime.now())

            if CLOSED:
                keyboard.press(Key.ctrl_l)
            else:
                keyboard.release(Key.ctrl_l)
            event = ""
            id = ""
            if BLINK:
                event = "blink"
                id = str(TOTAL)

            if CLOSED:
                event = "closed"
            if CLOSED == 0 and BLINK == 0:
                event = "open"

            filewriter.writerow(
                [id, datetime.now().strftime("%H:%M:%S"), event])

            img_post = f_detector.bounding_box(
                im, boxes_face, ['blinks: {}'.format(TOTAL), event])
        else:
            img_post = im

        # visualize
        end_time = time.time() - start_time
        FPS = 1/end_time
        cv2.putText(img_post, f"FPS: {round(FPS,3)}", (10, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_post, f"Blink MA: {len(BLINKLIST)}",
                    (10, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 40, 40), 2)
        cv2.imshow('blink_detection', img_post)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

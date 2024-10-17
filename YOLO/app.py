import random
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
# Load class list
class_list = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
                "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
                "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
                "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
                "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
                "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
                "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush","pillow"]
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in class_list]
model = YOLO("yolov8n.pt", "v8")
engine = pyttsx3.init()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
focal_length_x = 1000  # Focal length in pixels (x-axis)
focal_length_y = 1000  # Focal length in pixels (y-axis)
principal_point_x = 320  # Principal point x-coordinate (in pixels)
principal_point_y = 240  # Principal point y-coordinate (in pixels)
object_real_height = 1.0  
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    detected_objects = {}
    if len(detect_params) != 0:
        for box in detect_params[0].boxes:
            clsID, conf, bb = box.cls.numpy()[0], box.conf.numpy()[0], box.xyxy.numpy()[0]
            clsID, conf = int(clsID), float(conf)
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[clsID], 3)
            cv2.putText(frame, f"{class_list[clsID]} {conf:.3f}%", (int(bb[0]), int(bb[1]) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            object_height_pixels = bb[3] - bb[1]  # Height of the object bounding box in pixels
            distance = (focal_length_y * object_real_height) / object_height_pixels
            #bbox_width = bb[2] - bb[0]
            #bbox_height = bb[3] - bb[1]
            #bbox_size = (bbox_width + bbox_height) / 2 
            #distance = 1 / bbox_size
            object_center_x = (bb[0] + bb[2]) / 2  
            image_center_x = frame.shape[1] / 2  
            if object_center_x < image_center_x:
                direction = "left"
            elif object_center_x > image_center_x:
                direction = "right"
            else:
                direction = "center"
            print(class_list[clsID])
            print(direction)
            print(f"Distance to {class_list[clsID]}:", distance, "meters")
            detected_objects[class_list[clsID]] = [distance, direction]
    cv2.imshow("Object Detection", frame)
    print(detected_objects)
    objs = ''
    if detected_objects:
        for key,value in detected_objects.items():
            di = np.round(value[0])
            dr = value[1]
            n = key
            print(n)
            voice_text = n + " is " + str(di) +"meters away from "+dr +"of you "
            engine.say(voice_text)
            engine.runAndWait()
    else:
        engine.say("No object detected")
        engine.runAndWait()
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

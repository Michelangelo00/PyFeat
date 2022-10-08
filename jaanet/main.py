import csv
import json

from feat import Detector
from feat.utils import read_pictures
import cv2

#frame = cv2.imread("frame0.jpg")

frame = read_pictures(['frame0.jpg'])



detector = Detector(au_model = "jaanet")
detected_faces = detector.detect_faces(frame)

detected_landmarks = detector.detect_landmarks(frame, detected_faces)
#landmarks = detected_landmarks[0][0].tolist()
#print(detected_landmarks)

#hogs, new_lands = detector._batch_hog(
#        frames=frame, detected_faces=detected_faces, landmarks=detected_landmarks
#    )
detected_aus = detector.detect_aus(frame, detected_landmarks).flatten()
#print(detected_aus)
row = []
for element in detected_aus:
    #print(type(element))
    elemento = str(element.item())
    row.append(elemento+"   ")
    #print(type(elemento))
print(row)


#with open("frame0.txt", "a") as file:
#    for element in row:
#        file.write(str(element) + "\n")

filename = "frame0.csv"

with open(filename, 'w') as csvfile:
    
    csvwriter = csv.writer(csvfile)
    
    csvwriter.writerow(row)

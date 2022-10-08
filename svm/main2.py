import csv

from feat import Detector
from feat.utils import read_pictures


frame = read_pictures(['frame0.jpg'])

detector = Detector(au_model = "svm")
detected_faces = detector.detect_faces(frame)

detected_landmarks = detector.detect_landmarks(frame, detected_faces)

"""
detected_aus = detector.detect_aus(frame, detected_landmarks).flatten()

row = []
for element in detected_aus:
    #print(type(element))
    elemento = str(element.item())
    row.append(elemento+"   ")
    #print(type(elemento))
print(row)

filename = "frame0.csv"

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(row)

"""
import os

from feat import Detector
from feat.utils import read_pictures
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

face_model = "retinaface"
landmark_model = "mobilenet"
au_model = "jaanet"
emotion_model = "resmasknet"
detector = Detector(face_model = face_model, landmark_model = landmark_model, au_model = au_model, emotion_model = emotion_model)

start = time.time()

immagine = os.path.join("frame0.jpg")
frame = read_pictures(['frame0.jpg'])

f, ax = plt.subplots()

im = Image.open(immagine)
ax.imshow(im)

image_prediction = detector.detect_image(immagine)

print(image_prediction)

image_prediction.plot_detections()
plt.savefig("grafico.png")
end = time.time()

print(end-start)
plt.show()



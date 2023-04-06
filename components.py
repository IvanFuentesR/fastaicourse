from fastai.vision.all import *
from fastai.vision.utils import *
import matplotlib.pyplot as plt
import gradio as gr

learn2 = load_learner('components.pkl')

categories = ('cpu', 'gpu', 'motherboard')

def classify_images(img):
    pred,idx,probs = learn2.predict(img)
    print('values:')
    print(pred)
    print(idx)
    print(probs)
    return dict(zip(categories, map(float,probs)))

#imgCPU = plt.imread("cpu.jpeg")
#imgGPU = plt.imread("gpu.jpg")
imgMotherboard = plt.imread("motherboard.jpg")
#valuesCPU = classify_images(imgCPU)
#valuesGPU = classify_images(imgGPU)
valuesMotherboard = classify_images(imgMotherboard)

#print("CPU:")
#print(valuesCPU)

#print("GPU")
#print(valuesGPU)

print("Motherboard")
print(valuesMotherboard)

#print(learn)


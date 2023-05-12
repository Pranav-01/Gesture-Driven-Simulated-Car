import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from MobileNetV2_FC import Classifier
from dataset2 import landmarks
import cv2
import time
import mediapipe as mp
import pyautogui as pag

#add directory of the weights
model_path = "/home/arthur/Downloads/checkpoint_epoch22.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier()
model.to(device)
model.load_state_dict(torch.load(model_path,map_location=device))


def norm_points(points,width,height):

    x = 1/width
    y = 1/height

    pair = [x,y]
    arr = np.tile(pair,21)
    norm = arr*points
    # norm = norm.astype(int)
    return norm

def predict(image,points):
    key = ["left", "right", "backward", "forward", "stop"]
    # points = points.strip('[]')
    # points=points.split(',')
    points = np.array(points, dtype=np.float32)
    data_transforms = transforms.Compose([
    transforms.Resize((int(1920*0.3),int(1440*0.3))),
    transforms.ToTensor(),
    transforms.Normalize([0.5401, 0.5045, 0.4845], [0.2259, 0.2270, 0.2279])
    ])
    images = data_transforms(image)
    with torch.no_grad():
        model.eval()
        # images = data_transforms(image)
        image = images.unsqueeze(0)
        images = image.to(device=device, dtype=torch.float32)
        points = torch.from_numpy(points)
        points = points.to(device=device, dtype=torch.float32)
        points = points.unsqueeze(0)
        pred = model(points,images)
        pred = pred.cpu().detach().numpy().squeeze()
        output_probs = np.exp(pred)*100
        np.set_printoptions(suppress=True)
        # print(output_probs)
        output=pred.argmax()
        k = output.item()
        print("Predicted class label:", key[k])
        callback(k)
        
def callback(data):
    if data == 3:
        pag.typewrite(["i"])
    elif data == 1:
        pag.typewrite(["u"])
    elif data == 4:
        pag.typewrite(["k"])
    elif data == 0:
        pag.typewrite(["o"])
    elif data == 2:
        pag.typewrite([","])

prev = time.time()
vid = cv2.VideoCapture(0)

while(True):
    cur = time.time()
    ret, image = vid.read()
    # cv2.imshow('frame', image)

    if cur - prev > 1 and cur - prev < 1.4:
        prev = cur
        points, ptsimage = landmarks(image)
        # print("points")
        cv2.imshow('handpts',cv2.cvtColor(ptsimage, cv2.COLOR_RGB2BGR))
    
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert('RGB')
        width, height = image.size

        if len(points) > 0:
            points = norm_points(points, width, height)
            predict(image,points)
        else:
            continue
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
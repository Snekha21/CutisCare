import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_Color_Name(R,G,B):
    minimum = 10000
    csv = pd.read_csv("colors.csv")

    for i in range(len(csv)):
        print(csv["R"][i])
        d = abs(R- int(csv["R"][i])) + abs(G-int(csv["G"][i]))+ abs(B- int(csv["B"][i]))
        if(d<=minimum):
            minimum = d
            cname = csv.loc[i,"Name"]
    return cname
min_HSV = np.array([0, 58, 30], dtype = "uint8")
max_HSV = np.array([33, 255, 255], dtype = "uint8")
# Get pointer to video frames from primary device
image = cv2.imread("saddlebrown.png")
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)

skinHSV = cv2.bitwise_and(image, image, mask = skinRegionHSV)

cv2.imwrite("snek.jpg", np.hstack([image, skinHSV]))

import cv2
import numpy as np
import pandas as pd
 
img = cv2.imread("snek.jpg")
b,g,r = img[20,30]
b = int(b)
g = int(g)
r = int(r)
data = pd.read_csv("colors.csv")
cv2.rectangle(img,(20,20), (750,60), (b,g,r), -1)
text = get_Color_Name(r,g,b)+'R='+str(r)+'G='+ str(g)+'B='+ str(b)
cv2.putText(img, text,(50,50),2,0.8, (255,255,255),2,cv2.LINE_AA)
if(r+g+b>=600):
    cv2.putText(img, text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
print(text)

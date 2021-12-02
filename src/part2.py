import cv2
import numpy as np
from numpy.lib.function_base import append

lightDpath="data/light_directions.txt"
lightIpath="data/light_intensities.txt"
maskPath="data/mask.png"

def load_lightSources(n):
    with open(lightDpath,"r") as file:
        mat=np.empty((n,3),float)
        i=0
        for line in file:
            elements=line.split()
            print(elements[0])
            mat[i,0]=elements[0]
            mat[i,1]=elements[1]
            mat[i,2]=elements[2]
            i+=1
            
    return mat


def load_lightintensity(n):
    with open(lightIpath,"r") as file:
        mat=np.empty((n,3),float)
        i=0
        for line in file:
            elements=line.split()
            print(elements[0])
            mat[i,0]=elements[0]
            mat[i,1]=elements[1]
            mat[i,2]=elements[2]
            i+=1
            
    return mat


def load_objMask():
    img = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
    h,w=img.shape
    mask=np.zeros((h,w),np.uint16)
    for j in range(w):
        for i in range(h):
            if img[i,j]==255:
                mask[i,j]=1
    
    return mask


#light_source=load_lightSources(96)
mask=load_objMask()
import cv2
import numpy as np
from numpy.lib.function_base import append


def load_lightSources(n):
    with open("data/light_directions.txt","r") as file:
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
    with open("data/light_intensities.txt","r") as file:
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


light_source=load_lightSources(96)
print(light_source)
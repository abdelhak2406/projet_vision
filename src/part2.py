import cv2
import numpy as np

lightDpath="data/light_directions.txt"
lightIpath="data/light_intensities.txt"
maskPath="data/mask.png"

def load_lightSources():
    """Check the light_direction file and transform into a matrix 
        Object.
    """
    
    with open(lightDpath,"r") as file:
        n = len(file.readlines())
        i=0
        #put the file pointer into the beggining of the file(readlines puts it in the end).
        file.seek(0)
        mat=np.empty((n,3),float)
        for line in file:
            elements=line.split()
            mat[i,0]=elements[0]
            mat[i,1]=elements[1]
            mat[i,2]=elements[2]
            i+=1

    return mat


def load_lightintensity():
    """Check the light_intensities file and transform into a matrix 
        Object.
    """

    with open(lightIpath,"r") as file:
        n = len(file.readlines())
        i=0
        #put the file pointer into the beggining of thefile(readlines puts it in the end).
        file.seek(0)
        mat=np.empty((n,3),float)
        for line in file:
            elements=line.split()
            mat[i,0]=elements[0]
            mat[i,1]=elements[1]
            mat[i,2]=elements[2]
            i+=1
            
    return mat


def load_objMask():
    """"""
    img = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
    h,w=img.shape
    mask=np.zeros((h,w),np.uint16)
    for j in range(w):
        for i in range(h):
            if img[i,j]==255:
                mask[i,j]=1
    
    return mask


def main():
    mask=load_objMask()

if __name__ == "__main__":
    main()
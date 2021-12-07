import cv2
import numpy as np
from numpy.core.fromnumeric import transpose
from tqdm import tqdm
import math
from utilities import savePkl, openPkl
lightDpath="data/light_directions.txt"
lightIpath="data/light_intensities.txt"
maskPath="data/mask.png"
filenames_path="data/filenames.txt"

def load_lightSources():
    """Check the light_direction file and transform into a matrix 
        Object.

    Returns:
        matrix of (intensity B, intensity G, intensity R) for every
            row.
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
        mat_intensity=np.empty((n,3),float)
        for line in file:
            elements=line.split()
            mat_intensity[i,0]=elements[0]
            mat_intensity[i,1]=elements[1]
            mat_intensity[i,2]=elements[2]
            i+=1
            
    return mat_intensity


def load_objMask():
    """TODO:make sure it's correct and use normal not grayscale"""
    img = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
    h,w=img.shape
    mask=np.zeros((h,w),np.uint16)
    for j in range(w):
        for i in range(h):
            if img[i,j]==255:
                mask[i,j]=1
    
    return mask

def load_images(mat_intens,base_path="data/"):
    """Load all the images of the dataset. and execute
    all the steps mentionned in the project document(step4)"""
    h,w =  512, 612
    with open(filenames_path,"r") as file:
        n = len(file.readlines())
        file.seek(0)
        i = 0
        #the matrix that will contain all the flatten imamges
        img_mat =  np.zeros((n,h*w), np.float32)

        for picture_name in tqdm(file):
            picture_name = picture_name.strip()
            #import pdb;pdb.set_trace()
            img = read_img(base_path+picture_name)
            img = change_inter_img(img)
            img = div_img_intens(img,mat_intens[i])
            img = convert_grayscale(img)
            img_mat[i]= create_img_matrix(img)
            i = i + 1
 
    return img_mat

def change_inter_img(img):
    """Change image interval from uint16 [0,2^16 -1] to float32[0,1]

    Keyword arguments:
    img -- the image encoded with uint16 (rgb encoded not grayscale)


    Returns:
        the image with the new interval change! 
    """
    
    #empty img
    new_img = np.zeros(img.shape, np.float32)
    h,w,c = img.shape
    max_value = 2**16 - 1 
    #loop through the img
    for i in range(h):
        for j in range(w):
            for k in range(c):
                value = img[i,j,k]
                new_img[i,j,k] = img[i,j,k] / max_value #TODO: confirm it's the right operation

    return new_img


def div_img_intens(img, mat_intensity_line):
    """Divide every pixel in the img by the intensity of the source
    
    Keyword arguments:
    img -- the image encoded flot32
    mat_intensity_line  -- the intensity corresponding to the img 
                            in the mat_intensity
                            format [intensity B, intensity G, intensity R]

    Returns:
        the image with the new pixel changes
    """
    h,w,c = img.shape
    new_img = np.zeros(img.shape, np.float32)

    for i in range(h):
        for j in range(w):
            for k in range(c):
                new_img[i,j,k] =img[i,j,k] / mat_intensity_line[k]

    return new_img


def convert_grayscale(img):
    """Convert img to grayscale using the given formula.
    
    Keyword arguments:
    img -- the image in rgb (bgr) encoded in flot32
    """

    h,w,c = img.shape
    new_img = np.zeros((h,w), np.float32)
    # Splitting image into RGB channels:
    blue, green, red = cv2.split(img)
    for i in range(h):
        for j in range(w):
            new_img[i,j] = 0.3*red[i,j] + 0.59*green[i,j] +\
                            0.11*blue[i,j]

    return new_img


def read_img(img_path):
    """Read img in path

    Note: 
        h,w, channel img[h,w,chanl]
    """


    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    if img is not None:
        #import pdb; pdb.set_trace()
        return img        
    else:
        print("ERROR couldn't load the image at", img_path)
        exit(0)


def create_img_matrix(img):
    """take the img flatten it     Keyword arguments:

    img -- the image encoded flot32 with one channel
    """
    return img.flatten()
    

"""_________________________________ part2 __________________________________"""
def calcul_needle_map():
    #load mat_imgs and ligth sources
    obj_images = openPkl("mat_imgs.pkl", "out_objects/")
    light_sources = load_lightSources()
    
    #initializing a 3d array
    h,w =  512, 612
    line, col = 0, 0
    normals_mat = np.zeros((h,w,3), np.float32)

    #inverse of the light sources matrix
    inv_light_sources = np.linalg.pinv(light_sources)
    #get number of columns of mat_imgs
    nb_cols = obj_images.shape[1]
    #loop through the columns of mat_imgs
    for i in range(nb_cols):
        vector_e = obj_images[:,i]
        normal_vector = np.dot(inv_light_sources,vector_e)
        if col == 612:
            col = 0
            line += 1

        normals_mat[line, col, 0] = normal_vector[0]
        normals_mat[line, col, 1] = normal_vector[1]
        normals_mat[line, col, 2] = normal_vector[2]

        col += 1
    
    return normals_mat

def show_normals_in_img(filename, base_path = "out_objects/"):
    #load mask object and normals mat
    mask_obj = load_objMask()
    normals_mat = openPkl(filename, base_path)

    #initializing a 3d array
    h,w,c = normals_mat.shape
    new_img = np.zeros((h,w,c), np.uint8)
    
    for i in range(h):
        for j in range(w):
            if mask_obj[i,j] == 1:
                n = math.sqrt(normals_mat[i,j,0]**2 + normals_mat[i,j,1]**2 + normals_mat[i,j,2]**2)
                for k in range(c):
                    new_img[i,j,k] = int((((normals_mat[i,j,k]/n)+1)/2)*255)

    cv2.imshow('result', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    mask=load_objMask()
    mat_intens = load_lightintensity()
    mat_imgs = load_images(mat_intens=mat_intens)
    savePkl(mat_imgs,"mat_imgs.pkl","out_objects/")
    
    normals_mat = calcul_needle_map()
    savePkl(normals_mat,"normals_mat.pkl","out_objects/")
    """
    show_normals_in_img("normals_mat.pkl")
    

if __name__ == "__main__":
    main()
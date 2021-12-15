import cv2
import numpy as np
from scipy.sparse import csr_matrix
from numpy.core.fromnumeric import transpose
from tqdm import tqdm
import math
from utilities import savePkl, openPkl
import matplotlib.pyplot as plt
from scipy.linalg import orth
from scipy.fft import fft, ifft


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
    """take the img flatten it    
    
     Keyword arguments:

    img -- the image encoded flot32 with one channel
    """
    return img.flatten()
    

"""_________________________________ part2 __________________________________"""
def calcul_needle_map():
    #load mat_imgs and ligth sources
    obj_images = openPkl("mat_imgs.pkl", "out_objects/")
    light_sources = load_lightSources()
    obj_masques = load_objMask().flatten()

    
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
        if col == 612:
            col = 0
            line += 1

        if obj_masques[i] == 1:
            vector_e = obj_images[:,i]
            normal_vector = np.dot(inv_light_sources,vector_e)

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


"""_________________________________ part3 __________________________________"""
def depth_map_generation(obj_mask, normals_mat):
    """Generate the depth map of the object
    
    Keyword arguments:
    obj_mask -- the mask of the object
    normals_mat -- the normals of the object
    """

    #don't know what type of data to use

    h, w = obj_mask.shape

    index = np.zeros((h,w), np.uint8)

    nb_pixels = h*w
    object_pixel_rows = []
    object_pixel_cols = []

    #check if there is a function in np to do this
    for i in range(h):
        for j in range(w):
            if obj_mask[i,j] == 1:
                object_pixel_rows.append(i)
                object_pixel_cols.append(j)
    object_pixels = np.vstack((object_pixel_rows, object_pixel_cols))

    nb_pixels = object_pixels.shape[1]

    for d in range(nb_pixels):
        index[object_pixels[0,d], object_pixels[1,d]] = d
    
    
    m = np.zeros((2*nb_pixels, nb_pixels), np.uint8)
    b = np.zeros((2*nb_pixels, 1), np.float)
    #ça sert à quelque chose ?
    n = np.zeros((2*nb_pixels, 1), np.float)

    for d in range(nb_pixels):
        p_row = object_pixels[0,d]
        p_col = object_pixels[1,d]  
        nx = normals_mat[p_row, p_col, 0]
        ny = normals_mat[p_row, p_col, 1]
        nz = normals_mat[p_row, p_col, 2]


        if index[p_row, p_col+1] > 0 and index[p_row-1, p_col] > 0:
            m[2*d-1, index[p_row, p_col]] = 1
            m[2*d-1, index[p_row, p_col+1]] = -1
            b[2*d-1, 0] = nx/nz

            m[2*d, index[p_row, p_col]] = 1
            m[2*d, index[p_row-1, p_col]] = -1
            b[2*d, 0] = ny/nz
        
        elif index[p_row-1, p_col] > 0:
            f = -1
            if index[p_row, p_col+1] > 0:
                m[2*d-1, index[p_row, p_col]] = 1
                m[2*d-1, index[p_row, p_col+f]] = -1
                b[2*d-1, 0] = f*nx/nz

            m[2*d, index[p_row, p_col]] = 1
            m[2*d, index[p_row-1, p_col]] = -1
            b[2*d, 0] = ny/nz
        
        elif index[p_row, p_col+1] > 0:
            f = -1
            if index[p_row-f, p_col] > 0:
                m[2*d, index[p_row, p_col]] = 1
                m[2*d, index[p_row-f, p_col]] = -1
                n[2*d, 0] = f*ny/nz
            
            m[2*d-1, index[p_row, p_col]] = 1
            m[2*d-1, index[p_row, p_col+1]] = -1
            n[2*d-1, 0] = nx/nz
        
        else:
            f = -1
            if index[p_row, p_col+f] > 0:
                m[2*d-1, index[p_row, p_col]] = 1
                m[2*d-1, index[p_row, p_col+f]] = -1
                n[2*d-1, 0] = f*nx/nz
            
            f = -1
            if index[p_row-f, p_col] > 0:
                m[2*d, index[p_row, p_col]] = 1
                m[2*d, index[p_row-f, p_col]] = -1
                b[2*d, 0] = f*ny/nz
    
    print(m.shape)
    m = np.ma.masked_equal(m,0)
    print(m.shape)
    return "ok"
    #the problem is here
    x = np.linalg.lstsq(m, b)

    x = x - min(x)

    temp_shape = np.zeros((h,w), np.float32)
    for d in range(nb_pixels):
        p_row = object_pixels[0,d]
        p_col = object_pixels[1,d]
        temp_shape[p_row, p_col] = x[0,d]
    
    z = np.zeros((h,w), np.float32)
    for i in range(h):
        for j in range(w):
            z[i,j] = temp_shape[i,j]

    return z


def image_in_gray(image):
    h, w ,c = image.shape
    imageGray = np.zeros((h,w), np.float32)
    for y in range(h):  
        for x in range(w):
            imageGray[y,x]= image[y, x, 0]*0.11+ image[y, x, 1]*0.59+ image[y, x, 2]*0.3
    return imageGray

def calcul_3D(mask, image):
    #transforme l'image en niveau de gris
    imageG = image_in_gray(image)
    #imageG = (imageG+1)/2*255
    p = np.gradient(imageG, 1, axis=0)
    q = np.gradient(imageG, 1, axis=1)

    #initialisation de la matrice de profondeur
    z = np.zeros(imageG.shape, np.float32)
    h, w = imageG.shape

    #calcul de la profondeur au niveau de la premiere colonne
    for y in range(1,h):
        z[y,0] = z[y-1,0] - q[y,0]

    #calcul de la profondeur pour chaque ligne
    for y in range(h):
        for x in range(1,w):
            if mask[y,x] == 1 :
                z[y,x] = z[y, x-1] - p[y,x]


    ax = plt.axes(projection="3d")
    x, y = np.mgrid[0:512:512j, 0:612:612j]
    ax.plot_surface(x, y, z,cmap="Greys")
    plt.show()

    return z

    

   
def main():
    """
    mask=load_objMask()
    mat_intens = load_lightintensity()
    mat_imgs = load_images(mat_intens=mat_intens)
    savePkl(mat_imgs,"mat_imgs.pkl","out_objects/")
    
    normals_mat = calcul_needle_map()
    savePkl(normals_mat,"normals_mat.pkl","out_objects/")
    
    """
    mask = load_objMask()
    normals = openPkl("normals_mat.pkl","out_objects/")
    z = calcul_3D(mask, normals)
    
    

if __name__ == "__main__":
    main()
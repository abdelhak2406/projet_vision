"""Encode and decode the text inside a GRAYSCALE image (for now!)
"""
import cv2
import argparse
import numpy as np
import os

def parse_args():  
    '''
    my argumets!!
    '''
    parser = argparse.ArgumentParser(description="Image text encoding script.")
    # Path
    parser.add_argument('--impath', nargs='?',default='img/2000.png',
                        help='Image path of the image to code or decode depend on --action')
    parser.add_argument('--imcodepath', nargs='?',default='img/encoded_img.png',
                        help='Image path of the encoded image')
    #TODO: maybe supprimer!(il faut gerer la taille des donné!)
    parser.add_argument('--path_text', nargs='?',default='./tests/',
                    help='path to the textfile you want to send?')
    parser.add_argument('--text', nargs='?',default='tahya vision',
                    help='text to send in the image')
    parser.add_argument('--action', nargs='?',type=int ,default='1',
                        choices=[1,2],help='Encode or decode the image')

    return parser.parse_args()


def clear_bit(value, bit_index):
    """Set the bit in bit_index to 0.

    Keyword arguments:
    value --  the value as integer. ex: 0b11111111 .
    bit_index -- the bit in the value we want to set to 0.

    Returns:
        the value with the bit_index set to 0.

    Note:
        Thanks to RealPython for the function.
            Check the README references for details.
    """

    return value & ~(1 << bit_index)


def set_bit(value:int, bit_index):
    """Set the bit in bit_index to 1.

    Keyword arguments:
    value --  the value in binary. ex: 0b11111111 .
    bit_index -- the bit in the value we want to set to 1.

    Returns:
        the value with the bit_index set to 1.

    Note:
        Thanks to RealPython for the function.
            Check the README references for details.
    """

    return value | (1 << bit_index)


def get_bit(value:int, bit_index):
    """Get the bit in bit_index .

    Keyword arguments:
    value --  the value as an integer.
    bit_index -- the bit in the value we want to get to 1.

    Returns:
        the value of the bit at bit_index.

    Note:
        Thanks to RealPython for the function.
            Check the README references for details.
        - when the bit_index is greater than the size in bytes of the value, it doenes't throw any error!
    """

    return (value >> bit_index) & 1


def put_bit_in_value(value:int , bit_value, bit_index = 0):
    """Put a bit_value(0 or 1), in the bit_index of a value

    Keyword arguments:
    value --  the value as an integer( the pixel value).
    bit_index -- the bit in the value we want to set to bit_value.
    bit_value -- the value we want to set in the bit_index

    Returns:
        the value with the new bit value in bit_index changed.

    """
    if(bit_value == 1 ):
        value = set_bit(value, bit_index=bit_index)
    else:
        value =  clear_bit(value, bit_index=bit_index)
    return value


def encode_img(text:str, img):
    """Encode the text inside the image."""
    # check if the length is enough or not! 

    h , w= img.shape 
    max_length  = (w * h) / 8
    if( len(text) < max_length ):
        for i in range(len(text)):#used this cause i need i as a number later
            for j in range(8):
                char_i_j_th_bit = get_bit( ord(text[i])  , bit_index = j )

                _i = (i*8+j)//w
                _j = (i*8+j)%w
                img_pixel = img[_i][_j]

                img_pixel = put_bit_in_value(img_pixel, char_i_j_th_bit)
                img_pixel = img_pixel.astype(np.uint8) 
                img[_i][_j] = img_pixel
    else:
        exit("Text too big for the image!")
    return img


def decode_img(img):
    """Decode the text inside the image .
    
    Keyword arguments:
    img -- the image with the code inside

    Returns:
        the text encoded in the image
    
    Note:
        sometimes things work, just let them, just be happy about it!
    """
    img = img.flatten()
    text=""
    for i in range(0,len(img),8):
        car = chr(255)
        car_ascii= ord(car)
        for j in range(8):
            # get the first pixel
            last_bit_img = get_bit(img[i+j],0)

            car_ascii = put_bit_in_value(value = car_ascii, bit_value=last_bit_img ,bit_index =j) 

        car = chr(car_ascii)
        text = text + car 
        # i don't get why it works but it works, 
        # if we delete this we will see weird things!
        if(car=="\0"): 
            break
    print("------------------------------------------------")
    print("The text is: ",text)
    print("------------------------------------------------")


def read_img(img_path):
    """Read img in path"""

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img        
    else:
        exit("ERROR NO image or somthing!")


def save_img(img, img_path):
    """Save the image!"""

    # get the path witout the imgage name 
    path ="/".join(img_path.split("/")[:-1])
    if not os.path.exists(path):
        print("path:\"",path,"\"  doesnt exist so we created it")
        os.makedirs(path)

    cv2.imwrite(img_path, img)

def main():
    # read the image
    args = parse_args()
    img = read_img(args.impath)

    if args.action == 1:#encode
        img_encoded = encode_img(img=img,text=args.text)
        save_img(img_encoded, args.imcodepath)
    else:#decode
        decode_img(img)



if __name__ == '__main__':
    main()

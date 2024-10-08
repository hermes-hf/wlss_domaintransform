from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom

def get_luminance(img):
    res = 0.2126*img[:,:,0] + 0.7152*img[:,:,1] + 0.0722*img[:,:,2]
    return res

def rescale_matrix(matrix, L):
    """
    Rescale a given matrix to an L x L matrix using interpolation.
    
    Parameters:
    matrix (ndarray): Input matrix to be rescaled
    L (int): Desired size of the square matrix
    
    Returns:
    rescaled_matrix (ndarray): Rescaled L x L matrix
    """
    original_rows, original_cols = matrix.shape
    zoom_factors = (L / original_rows, L / original_cols)
    
    # Use scipy.ndimage.zoom to rescale the matrix
    rescaled_matrix = zoom(matrix, zoom_factors, order=1)  # order=3 for cubic interpolation
    
    return rescaled_matrix

def load_img_rgb(img_path, scale=0):
    image = Image.open(img_path)
    # Convert the image to a numpy array
    image_array = np.array(image)
    r = np.array(image_array[:,:,0], dtype=float)#/255
    g = np.array(image_array[:,:,1], dtype=float)#/255
    b = np.array(image_array[:,:,2], dtype=float)#/255

    if scale!=0:
        r = rescale_matrix(r,scale)
        g = rescale_matrix(g,scale)
        b = rescale_matrix(b,scale)
    
    N,M = np.shape(r)
    res = np.zeros((N,M,3))
    res[:,:,0] = r
    res[:,:,1] = g
    res[:,:,2] = b
    return res

def display_img(matrix):
    plt.imshow(matrix, cmap='gray')  # 'gray' for grayscale, use other colormaps if needed
    plt.colorbar()  # Optional: Add a colorbar to the side
    plt.title('Matrix as Image')  # Optional: Add a title to the image
    plt.show()
    return

def display_img_rgb(input_img):
    return Image.fromarray(np.uint8(input_img))
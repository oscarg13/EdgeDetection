import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.exposure import rescale_intensity
import cv2


class Sobel: 

    def __init__(self, image):
        self.image = image
        
   
    # Convert the RGB image to Gray Scale
    def convertToGray(self, image):
        return np.dot(image[:,:,:3], [0.299, 0.587, 0.114])
    
    # A method for kernel convolution with an image 
    def convole(self, image, kernel):
        (i_height, i_width) = image.shape[:2]
        (k_height, k_width) = kernel.shape[:2]

        padding = (k_width-1) // 2 
        padded_image = cv2.copyMakeBorder(image, padding ,padding,padding , padding , cv2.BORDER_REPLICATE)

        outputImage = np.zeros((i_height, i_width) , dtype='float32')
        
        for r in np.arange(padding , i_height + padding):
            for c in np.arange(padding , i_width + padding):
                win = padded_image[r-padding:r+padding+1, c-padding:c+padding+1]
                res = (win*kernel).sum()
                outputImage[r-padding , c - padding] = res
        
        outputImage = rescale_intensity(outputImage, in_range=(0,255))
        outputImage = (outputImage*255).astype("uint8")
        return outputImage



def main():
    
    # Create the laplacian operator kernel 
    laplacian = np.array([ [0, 1, 0], [1, -4, 1], [0, 1, 0] ])
    
    # read in the image
    src = mpimg.imread("SkyScraper.jpg")
    image = Sobel(src)
    
    # convert to gray scale
    grayscale = image.convertToGray(src)
    # convole the kernel with the image
    convole = image.convole(grayscale, laplacian)

    # display the resulting image 
    cv2.namedWindow('edge detection',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('edge detection', 600,600)
    output = cv2.imshow("edge detection", convole)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
main()
  


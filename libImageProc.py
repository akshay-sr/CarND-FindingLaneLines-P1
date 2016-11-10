import math
import cv2
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_extrapolate(image, color, thickness, lines):
    # print ('type(lines)=',type(lines),'lines=', lines)
    # print ('lines[0]=', lines[0])
    # print ('lines[0][0,0]=', lines[0][0,0])
    new_lines = []
    ''' Start with the beginning coords of the smallest in sorted order of line-segments
    for extrapolation'''
    new_line_coords = [lines[0][0,0], lines[0][0,1]] 
    for line in lines:
        for x1,y1,x2,y2 in line:
            midpoint_x = math.floor((x1+x2)/2)
            midpoint_y = math.ceil((y1+y2)/2)
            arr = np.array([new_line_coords[0], new_line_coords[1], midpoint_x, midpoint_y])
            new_lines.append([arr])
            new_line_coords = [midpoint_x, midpoint_y]

    ''' Add the last of the coords of the sorted line segments for extrapolation'''
    arr = np.array([new_lines[len(new_lines)-1][0][2],
                    new_lines[len(new_lines)-1][0][3],
                    lines[len(lines)-1][0,2],
                    lines[len(lines)-1][0,3]])

    new_lines.append([arr])
    for line in new_lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    ''' Draw lines without extrapolation '''
    # for line in lines:
        # for x1,y1,x2,y2 in line:
            # cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    ''' Draw lines with extrapolation '''
    left_lines = []
    right_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            ''' Since the coordinate system is top (0,0) slopes are reversed '''
            slope = (y2 - y1)/(x2 - x1)
            '''Threshold the slopes between 25 and 75 degrees'''
            if (slope < -0.4663 and slope > -3.7321):
                left_lines.append(line)
            elif (slope > 0.4663 and slope < 3.7321):
                right_lines.append(line)

    '''Eliminate passing lines that didn't make the threshold above'''
    if left_lines:
        draw_extrapolate(img, color, thickness, left_lines)
    if right_lines:
        draw_extrapolate(img, color, thickness, right_lines)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
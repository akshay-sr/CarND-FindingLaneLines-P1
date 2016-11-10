#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import libImageProc as libimg
from moviepy.editor import VideoFileClip

def process_image(image):
    '''printing out some stats and plotting'''
    #print('This image is:', type(image), 'with dimesions:', image.shape)
    #plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image
    #plt.show()

    '''Convert input image to grayscale'''
    gray = libimg.grayscale(image)
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    '''Define a kernel and apply Gaussian smoothing'''
    kernel_size = 5
    blur_gray = libimg.gaussian_blur(gray, kernel_size)
    # plt.imshow(blur_gray, cmap='gray')
    # plt.show()

    '''Define our parameters for Canny and apply'''
    low_threshold = 100
    high_threshold = 250
    edges = libimg.canny(blur_gray, low_threshold, high_threshold)

    '''Mask the canny edge output image to only contain the region of interest defined by the vertices'''
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(460, 330), (515, 330), (imshape[1],imshape[0])]], dtype=np.int32)

    masked_image = libimg.region_of_interest(edges, vertices)
    # plt.imshow(masked_image, cmap='gray')
    # plt.show()

    '''Run Hough on edge detected image. Output is a blank image with lines drawn on it'''
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_length = 5
    max_line_gap = 10
    lines = libimg.hough_lines(masked_image, rho, theta, threshold, min_line_length, max_line_gap)
    #print ('masked_image.shape= ', *image.shape)
    # plt.imshow(lines, cmap='gray')
    # plt.show()

    '''Create a "color" binary image to combine with line image from above'''
    #color_edges = np.dstack((edges, edges, edges)) 
    colorlines = libimg.weighted_img(lines, image)
    '''Flip the color channels from BGR to RGB as
    matplot and image files have the color channels switched
    If obj has shape (M,N,3), try swapping obj[:,:,0] with obj[:,:,2] '''
    result = np.fliplr(colorlines.reshape(-1,3)).reshape(colorlines.shape)

    return result

outputFile1 = "test_output/OutputSolidWhiteRight.jpg"
outputFile2 =  "test_output/OutputSolidWhiteCurve.jpg"
outputFile3 =  "test_output/OutputYellowCurve.jpg"
outputFile4 =  "test_output/OutputYellowCurve2.jpg"
outputFile5 =  "test_output/OutputSolidYellowLeft.jpg"
outputFile6 =  "test_output/OutputWhiteCarLaneSwitch.jpg"

inputFile1 = "test_images/solidWhiteRight.jpg"
inputFile2 = "test_images/solidWhiteCurve.jpg"
inputFile3 = "test_images/solidYellowCurve.jpg"
inputFile4 = "test_images/solidYellowCurve2.jpg"
inputFile5 = "test_images/solidYellowLeft.jpg"
inputFile6 = "test_images/whiteCarLaneSwitch.jpg"

outputFile = outputFile6
inputFile = inputFile6

#reading in an image from test_images
image = mpimg.imread(inputFile)


'''Process Video Input'''
# white_output = 'white.mp4'
# clip1 = VideoFileClip("solidWhiteRight.mp4")
# #NOTE: this function expects color images!!
# white_clip = clip1.fl_image(process_image)
# white_clip.write_videofile(white_output, audio=False)

# yellow_output = 'yellow.mp4'
# clip2 = VideoFileClip('solidYellowLeft.mp4')
# yellow_clip = clip2.fl_image(process_image)
# yellow_clip.write_videofile(yellow_output, audio=False)

'''Process Image Input'''
colorlines = process_image(image)
cv2.imwrite(outputFile, colorlines, [cv2.IMWRITE_JPEG_QUALITY, 100])
plt.imshow(colorlines)
plt.show()

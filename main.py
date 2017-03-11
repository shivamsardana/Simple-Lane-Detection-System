import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def grayscale(img):

    """Applies the Grayscale transform
    This will return an image with only one color channel
    """
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Blur kernel"""

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    
    # defining a blank mask to start with
    
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    averaging 
    &
    extrapolating 
    lines points achieved.
    """
    
    if len(img.shape) == 2:  # grayscale image -> make a "color" image out of it
        img = np.dstack((img, img, img))

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 >= 0 and x1 < img.shape[1] and \
                            y1 >= 0 and y1 < img.shape[0] and \
                            x2 >= 0 and x2 < img.shape[1] and \
                            y2 >= 0 and y2 < img.shape[0]:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            else:
                print('BAD LINE (%d, %d, %d, %d)' % (x1, y1, x2, y2))





def weighted_img(img, initial_img, alpha=0.8, beta=1., lamda=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, lamda)





def preprocess(img):
    """ Preprocess the input image """
    v=grayscale(img)


    # Apply Gaussian Blur to reduce the noise in edge detection
    kernel_size = 5
    out = gaussian_blur(v, kernel_size)

    # plt.subplot(2,2,1)
    # show_img(out)
    return out


def apply_canny(img):
    """ Applies the Canny edge detector to the input image """
    # Apply Canny edge detector
    low_threshold = 55
    high_threshold = 140

    out_img = canny(img, low_threshold, high_threshold)
    # show_img(out_img)
    return out_img

def select_region_of_interest(img):

'''
Definig ROI, you can change ROI according to your images and it is passing manually, if you want to analyse points, use MATLAB tool 
function, its good to analyse points.
'''

    h = 20
    v1 = (0 + h, img.shape[0])
    v2 = (img.shape[1] / 3.2, img.shape[0] / 2)
    v3 = (img.shape[1] / 1.64, img.shape[0] / 2)
    v4 = (img.shape[1] / 1.28, img.shape[0])
    '''
    h = 20
    k = 1.35
    v1 = (0 + h, img.shape[0])
    v2 = (img.shape[1] / 1.85  +h, img.shape[0] / 2)
    v3 = (img.shape[1] / 1.98*k  -h, img.shape[0] / 2)
    v4 = (img.shape[1] /1.35, img.shape[0])
    '''
    return region_of_interest(img, np.array([[v1, v2, v3, v4]], dtype=np.int32))
    


def extract_edges(img):

    # Get edges using the Canny edge detector
    
    img_canny = apply_canny(img)
    return select_region_of_interest(img_canny)


def detect_lines(img_canny_masked):

    """ Runs the Hough transform to detect lines in the input image"""
    # Apply HoughLines to extract lines
    
    rho_res = .1 # [pixels]
    theta_res = np.pi / 180.  # [radians]
    threshold = 7  # [# votes]
    min_line_length = 11 # [pixels]
    max_line_gap = 1  # [pixels]
    lines = cv2.HoughLinesP(img_canny_masked, rho_res, theta_res, threshold, np.array([]),
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def fitLine(line_points):

    """ Given 2 points (x1,y1,x2,y2), compute the line equation
    y = mx + b"""
    
    x1 = line_points[0]
    y1 = line_points[1]
    x2 = line_points[2]
    y2 = line_points[3]

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return (m, b)


def extract_lanemarkings(img_shape, lines):

    """ Given a list of lines (detected by the Probabilistic Hough transform),
    average and extrapolate them in order to come up with 2 single
    lines, corresponding to the left and right lanemarkings """
    
    # For each line segment
    slope_min = 0.8
    slope_max = 2.4

    m1 = np.array([])
    b1 = np.array([])

    m2 = np.array([])
    b2 = np.array([])

    y_min = img_shape[0]

    for line_points in lines:
        # Fit to line equation (m, b)
        (m, b) = fitLine(line_points)

        # Filter line by slope
        if abs(m) > slope_min and abs(m) < slope_max:
            y_min = min(y_min, line_points[1])
            y_min = min(y_min, line_points[3])

            # Separate into left/right using the sign of the slope
            if (m > 0):
                m1 = np.append(m1, m)
                b1 = np.append(b1, b)
            else:
                m2 = np.append(m2, m)
                b2 = np.append(b2, b)


    # Average the two main lines
    
    m1 = np.mean(m1)
    b1 = np.mean(b1)

    m2 = np.mean(m2)
    b2 = np.mean(b2)

    # Compute the crossing (x,y) point in the image
    x_cross = (b2 - b1) / (m1 - m2)
    y_cross = m1 * x_cross + b1

    # End point of the line: at most the crossing point
    y_end = max(y_cross, y_min)

    # Compute the (x) coordinate where the line crosses the
    # bottom edge of the image
    y1 = img_shape[0] - 1
    x1 = (y1 - b1) / m1
    y2 = img_shape[0] - 1
    x2 = (img_shape[0] - b2) / m2

    x_end1 = (y_end - b1) / m1
    x_end2 = (y_end - b2) / m2

    return np.array([[[x1, y1, x_end1, y_end]], [[x2, y2, x_end2, y_end]]]).astype(int)


def overlay_lanemarkings(img, lanemarkings):
    """ Draws the lines on top of the image img """
    # Create a black image with red lanemarkings
    img_lines = np.copy(img) * 0
    draw_lines(img_lines, lanemarkings, color=[255, 0, 0], thickness=10)

    # Blend the original image with the previous one
    img_out = weighted_img(img_lines, img)
    return img_out


def pipeline(img_original):

    """
    Process the input image 'img' and outputs an annotated version of it,
    where the left and right lane markings are detected.
    """
    
    # Pre-process
    
    img = preprocess(img_original)

    # Extract edges
    
    img_edges = extract_edges(img)

    # Detect lines
    
    lines = detect_lines(img_edges)
    img_lines = np.copy(img_original)
    draw_lines(img_lines, lines)

    # Extract left and right lanemarkings from the lines
    
    lanemarkings = extract_lanemarkings(img.shape, lines.squeeze())

    # Produce output
    img_out = overlay_lanemarkings(img_original, lanemarkings)
    # img_out = img_lines
    return img_out

i=0
cap=cv2.VideoCapture('um_%06d.png')
while(cap.isOpened()):
        ret, img = cap.read()
        ht, wd, dp = img.shape

        imgout = pipeline(img)
        a='.png'
        b=str(i)+a
        i=i+1
        cv2.imwrite(b,imgout)
        cv2.waitKey()
        
#Passing single image
'''
img=cv2.imread('um_000000.png')

imgout=pipeline(img)
cv2.imshow('output',imgout)
cv2.waitKey()
'''

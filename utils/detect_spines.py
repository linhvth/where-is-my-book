import cv2
import numpy as np
from utils import *
import os
import pandas as pd

from utils.utils import display_plt

def get_single_self():
    return

def export_spine_img(global_path, img_path, book_spines):
    """
    Export image for each book spine. 
    The exported images are written in /book-spines/ folder
    """
    name = img_path.split('/')[1].split('.')[0]
    curr_i = 0
    folder_spines = f"{global_path}/book-spines"
    os.makedirs(folder_spines, exist_ok=True) 

    for spine in book_spines:
        if spine.shape[1] == 0:
            curr_i += 1
            continue

        spine_path = f"{global_path}/book-spines/{name}-spine{str(curr_i)}.jpg"
        cv2.imwrite(spine_path, spine)
        curr_i += 1


def get_spine_loc(img, dividers):
    spine_location = pd.DataFrame(columns=['x1', 'x2', 'text'])
    div_start = 0
    div_end = 1
    h = img.shape[0]
    w = img.shape[1]

    while div_start < len(dividers) - 1:
        x1 = int(dividers[div_start][0])
        x2 = int(dividers[div_end][0])

        if (div_start == 0) and (x1 != 0): # first (potential) spine
            spine_location.loc[len(spine_location)] = [0, x1, '']
        
        single_spine = [x1, x2, '']
        spine_location.loc[len(spine_location)] = single_spine
        
        div_start += 1
        div_end += 1

        if div_end == len(dividers): # last (potential) spine
            spine_location.loc[len(spine_location)] = [x2, w, '']

    return spine_location


def get_book_spines(img, dividers):
    book_spines = []
    div_start = 0 # index of the first divider for a single spine
    div_end = 1 # index of the second divider for a single spine
    h = img.shape[0]
    w = img.shape[1]

    while div_start < len(dividers)-1:
        x1 = int(dividers[div_start][0])
        x2 = int(dividers[div_end][0])

        if (div_start == 0) and (x1 != 0): # first (potential) spine
            book_spines.append(img[0:h, 0:x1])
        
        
        single_spine = img[0:h, x1:x2]
        book_spines.append(single_spine)
        
        div_start += 1
        div_end += 1

        if div_end == len(dividers): # last (potential) spine
            book_spines.append(img[0:h, x2:w])


    return book_spines

def get_divider_lines(img, lines):
    draft = line_sifting(img, lines)
    dividers = get_important_lines(draft)
    # img_show = draw_vertical1(img, houghlines)
    return dividers

def get_vertical_lines(img_gray, min_vertical=0.15):
    """
    This function aims to highlight/extract potential vertical lines only.
    This is adpated from OpenCV tutorial:
    https://docs.opencv.org/3.4/dd/dd7/tutorial_morph_lines_detection.html
    """
    # Apply adaptiveThreshold at the bitwise_not of gray
    gray = cv2.bitwise_not(img_gray)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    vertical = np.copy(bw)

    # Specify size on vertical axis
    max_length = vertical.shape[0]    
    verticalsize =  round(max_length * min_vertical)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    
    # Smooth line
    edges = cv2.adaptiveThreshold(vertical, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 3, -2)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel)

    smooth = np.copy(vertical)
    smooth = cv2.GaussianBlur(smooth, (3, 3), 0)

    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]
    
    # display_plt(img_gray, vertical, '', '')

    return vertical

def houghlines(vertical_lines, min_line_len=200, max_line_gap=10, 
               rho=1, theta=np.pi/180, threshold=50):
    """
    Apply HoughLineP to fine lines.
    """
    linesP = cv2.HoughLinesP(image=vertical_lines, rho=rho, 
                            theta=theta, threshold=threshold, 
                            lines=None, 
                            minLineLength=min_line_len, 
                            maxLineGap=max_line_gap)
    
    simple_form = []
    for i in range(len(linesP)):
        simple_form.append(linesP[i][0])

    return simple_form

def eliminate_cross(lines, circles, tolerance=10): # not good yet
    curr_x_c = 0
    curr_x_l = 0
    final_lines = lines.copy()

    for l in lines:
        x_l = l[0]
        for c in circles:
            x_c = c[0]
            radius = c[2]//2
            if x_l not in range(x_c - radius, x_c + radius):
                continue
            else:
                if x_l in range(x_c - radius + tolerance, x_c + radius - tolerance):
                    final_lines.remove(l)
    
    print(len(final_lines))
    return final_lines

def get_important_lines(lines):
    """
    
    """
    i = 0
    j = 0
    max_index = len(lines) - 1

    dividers = []
    while i < max_index:
        if j >= max_index: break # out of index, break loop
        
        j = i + 1 
        dividers.append(lines[i])

        while j < max_index:
            # if 2 separated lines are too close
            if lines[j][0] - lines[i][0] < 70:
                j = j + 1 # ignore the current line
            else: # this is a divider
                i = j
                break
    print(len(dividers))
    return dividers

def line_sifting(img, lines, min_length=0.15): #bug
    """
    
    """
    h = img.shape[0]
    strong_lines = []
    print("before: ", len(lines))


    if lines is not None:
        for line in lines:
            x = line[0] # x is the same for both points
            y1 = line[1]
            y2 = line[3]
            length_y = abs(y1 - y2)

            if (length_y > h*min_length):
                strong_lines.append([x, length_y])
                
    strong_lines.sort() # sort by x-axis
    print(len(strong_lines))
    return strong_lines

def blue_circles(img, img_hsv): # not optimal yet, dont use for now
    threshold = img.shape[0]//2

    frame = img.copy()
    frame_gau_blur = cv2.GaussianBlur(frame, (3, 3), 0)
    
    # the range of blue color in HSV
    lower_blue = np.array([110,140,0])
    upper_blue = np.array([150,255,255])
    
    # getting the range of blue color in frame
    blue_range = cv2.inRange(img_hsv, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(frame_gau_blur,frame_gau_blur, mask=blue_range)
    blue_s_gray = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)
    canny_edge = cv2.Canny(blue_s_gray, 200, 255)
    # display_plt(canny_edge, img, '', '')
    
    # applying HoughCircles
    draft_circles = cv2.HoughCircles(canny_edge, cv2.HOUGH_GRADIENT, dp=1, minDist=170, \
                               param1=5, param2=15, minRadius=10, maxRadius=80)
    
    circles = []
    if draft_circles is not None:
        draft_circles = np.uint16(np.around(draft_circles))
        for i in draft_circles[0, :]:
            center = (i[0], i[1])
            if center[1] < threshold:
                circles.append(i)
                
                # circle center
                cv2.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(frame, center, radius, (255, 0, 0), 5)
    
    display_plt(img, frame, '', '')
    
    return circles



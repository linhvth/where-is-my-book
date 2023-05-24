import cv2
from utils.pre_processing import *
from utils.detect_spines import *
from utils.utils import *
from utils.text_detection import *
import glob
import os
import pandas as pd

def book_spines(img, img_gray):
    vertical_lines = get_vertical_lines(img_gray, min_vertical=0.1)
    linesP = houghlines(vertical_lines)
    dividers = get_divider_lines(img, linesP)
    spines = get_book_spines(img, dividers)
    location = get_spine_loc(img, dividers)

    return spines, location

if __name__ == '__main__':
    # images = glob.glob('image/*.jpg')
    images = ['image/fulbright-lib-3.jpg']
    global_path = os.getcwd()


    for img_path in images:
        img, img_RGB, img_hsv, img_gray = load_img(img_path)
        spines, spine_loc = book_spines(img, img_gray)
        export_spine_img(global_path,img_path, spines)
        
        # if have new image
        text_recog(img, spine_loc, method='easyorc')
        spine_loc = spine_loc.dropna()
        # spine_loc.to_csv('fulbright-lib-1_text_easyorc.csv', )

        isUse = True
        while isUse:
            query = input('What book do you want to find?:')
            best_match = matching(query, spine_loc)
            if best_match is None:
                print("We cannot find your book.")
            else:
                drawing_roi(img, best_match)
            confirm = input('Do you want to find more books?')
            if confirm == 'no':
                isUse = False
        
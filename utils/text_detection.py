# import the necessary packages
import pytesseract
import cv2
import easyocr
import numpy as np
import pandas as pd

BLOCKLIST = "0123456789`[]!@#$%^&*()_+-=:}{<>?"

def img_roi(img, p_top=0.25, p_bot=0.8):
    top = round(img.shape[0]*p_top)
    bot = round(img.shape[0]*p_bot)

    roi = img[top:bot, :]

    return roi

def text_recog_tesseract(img):
    img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    result = pytesseract.image_to_string(img_rot)

    if result == '':
        result = pytesseract.image_to_string(img)

    return result

def text_recog_easyorc(img):
    img_rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) 
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(img_rot, blocklist=BLOCKLIST)
    
    if results == []:
        results = reader.readtext(img, blocklist=BLOCKLIST)

    results = pd.DataFrame(results)

    if results.shape[0] != 0:
        results = results[results.iloc[:,2] > 0.3]

        text = ''
        col_text = results.iloc[:, 1]
        for t in col_text:
            text += ' ' + t.lower()

        return text

def text_recog(img, spine_loc, method):
    img_crop = img_roi(img.copy())
    h = img_crop.shape[0]

    if method == 'easyorc':
        for i, row in spine_loc.iterrows():
            x1 = row['x1']
            x2 = row['x2']
            roi = img_crop[0:h, x1:x2]
            text = text_recog_easyorc(roi)
            spine_loc.at[i, 'text'] = text
    
    elif method == 'tesseract':
        for i, row in spine_loc.iterrows():
            x1 = row['x1']
            x2 = row['x2']
            roi = img_crop[0:h, x1:x2]
            text = text_recog_tesseract(roi)
            spine_loc.at[i, 'text'] = text


def matching(query, spines_loc):
    query = query.lower().split(' ')
    best_score = 0
    best_match = None

    for i, r in spines_loc.iterrows():
        text = r['text']
        lst = text.split(' ')
        score = 0
        solid_score = 0
        soft_score = 0
        
        for w in query:
            if w in lst:
                solid_score += 1
            else:
                for t in lst:
                    if (len(t) > 1) and (t in w):
                        soft_score += 1

        if solid_score != 0:
            score = solid_score*0.7 + soft_score*0.3    
            if score > best_score:
                best_score = score
                best_match = r

    return best_match

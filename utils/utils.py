import cv2
import matplotlib.pyplot as plt

def display_plt(imgL, imgR, titleL, titleR):
    plt.figure(figsize=(15, 30))
    plt.subplot(121)
    plt.title(titleL)
    plt.imshow(imgL, cmap='gray')
    plt.subplot(122)
    plt.title(titleR)
    plt.imshow(imgR, cmap='gray')

def draw_dividers(img, dividers):
    """
    This function aims to draw dividers on the input image
    Params:
    -   img: color img is prefered
    -   dividers: list of lines that divide each book spine
    """
    new_img = img.copy()
    w = img.shape[1]

    for line in dividers:
        x = line[0]
        pt1 = (x, 0)
        pt2 = (x, w)
        cv2.line(new_img, pt1, pt2, (255, 0, 0), 4, cv2.LINE_AA)

    return new_img

def drawing_roi(img, match, ipynb=False):
    img_copy = img.copy()
    h = img_copy.shape[0]
    x1 = match['x1']
    x2 = match['x2']
    pt1 = (x1, round(h*0.2))
    pt2 = (x2, round(h*0.95))
    cv2.rectangle(img_copy, pt1, pt2, [255,0,0], 15)

    if ipynb:
        return img_copy
    else:
        cv2.namedWindow('Where is my book?', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Where is my book?', img_copy)
        cv2.resizeWindow('Where is my book?', 800, 600)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

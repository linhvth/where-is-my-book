import cv2

def load_img(path):
    img = cv2.imread(path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    return img, img_RGB, img_hsv, img_gray

def blur(img):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    return img_blur


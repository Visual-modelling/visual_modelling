# Identify the coordinates of the ball in the image
import numpy as np
import cv2
from PIL import Image

def get_circle(image):
    '''Identify location of center'''
    image = image.numpy()[0]
    assert len(image.shape) == 2
    minDist = 5
    param1 = 5
    param2 = 10  # smaller value-> more false circles
    minRadius = 4
    maxRadius = minRadius+1
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    cur_x = None
    cur_y = None
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if image[y, x] > 20:
                cur_x = x
                cur_y = y
    else:
        #print("No circles")
        pass
    return (cur_x, cur_y)

def calculate_metric(img_1, img_2):
    '''Get euclidian distance between circle centers'''
    x1, y1 = get_circle(img_1)
    x2, y2 = get_circle(img_2)
    if x1 is None or x2 is None:
        return None
    return np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))


if __name__ == "__main__":
    print(calculate_metric("sample_output/Frame 2 (100ms) (replace).png", "sample_output/Frame 4 (100ms) (replace).png"))

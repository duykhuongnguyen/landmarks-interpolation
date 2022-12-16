import numpy as np                       
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt          

import cv2


if __name__ == '__main__':
    img = cv2.imread("sample/40544470_3.jpg")

    with open("sample/40544470_3.txt", "r") as f:
        lines = f.readlines()
        eye_points = [67 + i for i in range(0, 8)] + [76 + i for i in range(8)]
        left_eye_points = [67 + i for i in range(0, 8)]
        right_eye_points = [76 + i for i in range(8)]

        connected = []
        for i in range(len(left_eye_points)):
            if i == len(left_eye_points) - 1:
                connected.append([left_eye_points[i], left_eye_points[0]])
            else:
                connected.append([left_eye_points[i], left_eye_points[i + 1]])
        
        for i in range(len(right_eye_points)):
            if i == len(right_eye_points) - 1:
                connected.append([right_eye_points[i], right_eye_points[0]])
            else:
                connected.append([right_eye_points[i], right_eye_points[i + 1]])

        for i in eye_points:
        # for i in range(1, len(lines)):
            text = lines[i][:-1]
            x, y = text.split(" ")
            x, y = int(float(x)), int(float(y))

            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        for line in connected:
            text1 = lines[line[0]][:-1]
            text2 = lines[line[1]][:-1]

            x1, y1 = text1.split(" ")
            x1, y1 = int(float(x1)), int(float(y1))
            
            x2, y2 = text2.split(" ")
            x2, y2 = int(float(x2)), int(float(y2))

            # cv2.circle(img, (x, y), 1, (0, 0, 255))
            cv2.line(img, [x1, y1], [x2, y2], [0, 0, 255], 1) 

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", img)
    cv2.waitKey(0)

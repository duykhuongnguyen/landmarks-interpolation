import numpy as np                       
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import find_peaks
import matplotlib.pyplot as plt          
                                         
import cv2                               


def quadratic_spline_roots(spl):
    roots = []
    knots = spl.get_knots()
    for a, b in zip(knots[:-1], knots[1:]):
        u, v, w = spl(a), spl((a+b)/2), spl(b)
        t = np.roots([u+w-2*v, w-u, 2*v])
        t = t[np.isreal(t) & (np.abs(t) <= 1)]
        roots.extend(t*(b-a)/2 + (b+a)/2)
    return np.array(roots)


fname = "40544470_3"
img = cv2.imread(f"sample/{fname}.jpg")

with open(f"sample/{fname}.txt", "r") as f:
    lines = f.readlines()
    eye_points = [67 + i for i in range(0, 8)] + [76 + i for i in range(8)]
    left_eye_points = [67 + i for i in range(0, 8)]
    right_eye_points = [76 + i for i in range(8)]

    # high_left = [67, 68, 69, 70, 71]
    high_left = [67, 68, 70, 71]
    # low_left = [71, 72, 74, 67]
    low_left = [67, 74, 72, 71]
    high_right = [76, 77, 79, 80]
    # low_right = [80, 81, 83, 76]
    low_right = [76, 83, 81, 80]
    eye_points = high_left + low_left + high_right + low_right

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
                                                                        
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    high_left_cords_x, high_left_cords_y = [], []
    for i in high_left:
        text = lines[i][:-1]
        x, y = text.split(" ")
        x, y = int(float(x)), int(float(y))

        high_left_cords_x.append(x)
        high_left_cords_y.append(y)
    
    cs1 = InterpolatedUnivariateSpline(high_left_cords_x, high_left_cords_y, k=3)
    xs1 = np.linspace(high_left_cords_x[0], high_left_cords_x[-1])

    cr_pts = quadratic_spline_roots(cs1.derivative())
    cr_pts = np.append(cr_pts, (high_left_cords_x[1], high_left_cords_x[2]))  # also check the endpoints of the interval
    cr_vals = cs1(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals)
    print("Maximum value {} at {}\nMinimum value {} at {}".format(cr_vals[max_index], cr_pts[max_index], cr_vals[min_index], cr_pts[min_index]))
    # x_new = int(cr_pts[min_index])
    x_new = (high_left_cords_x[1] + high_left_cords_x[2]) // 2
    y_new = int(cs1(x_new))
    cv2.circle(img, (x_new, y_new), 1, (0, 0, 255), -1)

    low_left_cords_x, low_left_cords_y = [], []
    for i in low_left:
        text = lines[i][:-1]
        x, y = text.split(" ")
        x, y = int(float(x)), int(float(y))

        low_left_cords_x.append(x)
        low_left_cords_y.append(y)

    cs2 = InterpolatedUnivariateSpline(low_left_cords_x, low_left_cords_y, k=3)
    xs2 = np.linspace(low_left_cords_x[0], low_left_cords_x[-1])

    cr_pts = quadratic_spline_roots(cs2.derivative())
    cr_pts = np.append(cr_pts, (low_left_cords_x[1], low_left_cords_x[2]))  # also check the endpoints of the interval
    cr_vals = cs2(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals)
    print("Maximum value {} at {}\nMinimum value {} at {}".format(cr_vals[max_index],cr_pts[max_index], cr_vals[min_index], cr_pts[min_index]))
    # x_new = int(cr_pts[max_index] + 2)
    x_new = (low_left_cords_x[1] + low_left_cords_x[2]) // 2
    y_new = int(cs2(x_new))
    print(x_new, cr_pts[max_index], y_new, cr_vals[max_index])
    cv2.circle(img, (x_new, y_new), 1, (0, 0, 255), -1)

    high_right_cords_x, high_right_cords_y = [], []
    for i in high_right:
        text = lines[i][:-1]
        x, y = text.split(" ")
        x, y = int(float(x)), int(float(y))

        high_right_cords_x.append(x)
        high_right_cords_y.append(y)
    
    cs3 = InterpolatedUnivariateSpline(high_right_cords_x, high_right_cords_y, k=3)
    xs3 = np.linspace(high_right_cords_x[0], high_right_cords_x[-1])

    cr_pts = quadratic_spline_roots(cs3.derivative())
    cr_pts = np.append(cr_pts, (high_right_cords_x[1], high_right_cords_x[2]))  # also check the endpoints of the interval
    cr_vals = cs3(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals)
    print("Maximum value {} at {}\nMinimum value {} at {}".format(cr_vals[max_index], cr_pts[max_index], cr_vals[min_index], cr_pts[min_index]))
    # x_new = int(cr_pts[min_index])
    x_new = int((high_right_cords_x[1] + high_right_cords_x[2]) / 2)
    y_new = int(cs3(x_new))
    cv2.circle(img, (x_new, y_new), 1, (0, 0, 255), -1)

    low_right_cords_x, low_right_cords_y = [], []
    for i in low_right:
        text = lines[i][:-1]
        x, y = text.split(" ")
        x, y = int(float(x)), int(float(y))

        low_right_cords_x.append(x)
        low_right_cords_y.append(y)

    cs4 = InterpolatedUnivariateSpline(low_right_cords_x, low_right_cords_y, k=3)
    xs4 = np.linspace(low_right_cords_x[0], low_right_cords_x[-1])

    cr_pts = quadratic_spline_roots(cs4.derivative())
    cr_pts = np.append(cr_pts, (low_right_cords_x[1], low_right_cords_x[2]))  # also check the endpoints of the interval
    cr_vals = cs4(cr_pts)
    min_index = np.argmin(cr_vals)
    max_index = np.argmax(cr_vals)
    print("Maximum value {} at {}\nMinimum value {} at {}".format(cr_vals[max_index],cr_pts[max_index], cr_vals[min_index], cr_pts[min_index]))
    # x_new = int(cr_pts[max_index] + 2)
    x_new = int((low_right_cords_x[1] + low_right_cords_x[2]) / 2)
    y_new = int(cs4(x_new))
    print(x_new, cr_pts[max_index], y_new, cr_vals[max_index])
    cv2.circle(img, (x_new, y_new), 1, (0, 0, 255), -1)
    
    # for line in connected:                                                  
    #     text1 = lines[line[0]][:-1]                                         
    #     text2 = lines[line[1]][:-1]                                         
                                                                        
    #     x1, y1 = text1.split(" ")                                           
    #     x1, y1 = int(float(x1)), int(float(y1))
    #     print(x1, y1)
                                                                        
    #     x2, y2 = text2.split(" ")                                           
    #     x2, y2 = int(float(x2)), int(float(y2))                             
                                                                        
        # cv2.circle(img, (x, y), 1, (0, 0, 255))                           
    #     cv2.line(img, [x1, y1], [x2, y2], [0, 0, 255], 1)                   

# show the output image with the face detections + facial landmarks
# cv2.imshow("Output", img)
# cv2.waitKey(0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.plot(xs1, cs1(xs1), c="blue", linewidth=0.5)
plt.plot(xs2, cs2(xs2), c="blue", linewidth=0.5)
plt.plot(xs3, cs3(xs3), c="blue", linewidth=0.5)
plt.plot(xs4, cs4(xs4), c="blue", linewidth=0.5)
plt.axis('off')
plt.tight_layout()
plt.savefig(f"results/{fname}_curve.png", dpi=400)
plt.show()

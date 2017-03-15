import cv2
import math


def new_coordinates(point_one, point_two):
    print(("x1", point_one[0], "y1", point_one[1]), ("x2", point_two[0], "y2", point_two[1]))
    distance = math.sqrt((point_two[0] - point_one[0]) ** 2 + (point_two[1] - point_one[1]) ** 2)
    print("distance between point one and two", distance)

    slope = (point_two[1] - point_one[1]) / (point_two[0] - point_one[0])
    print("Slope", slope)

    angle = math.atan(slope)
    print("angle", angle)

    a = math.sin(angle) * (distance/5)
    print("a", a)
    b = math.cos(angle) * (distance/5)
    print("b", b)

    x_a = point_one[0] - a
    y_b = point_one[1] - b

    print("New points", (int(x_a), int(y_b)))
    new_distance = math.sqrt((int(x_a) - point_two[0]) ** 2 + (int(y_b) - point_two[1]) ** 2)
    print("new distance", new_distance)
    return int(x_a), int(y_b)

new_co = new_coordinates((400,400), (600,200))

img = cv2.imread('test_images/solidWhiteRight.jpg')
cv2.line(img, (400,400), (600,200), [0,255,0], 7)
cv2.line(img, (400,400), new_co, [204,255,204], 7)
cv2.imshow('image', img)
cv2.waitKey(0)
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
import random
import gradio as gr
from ShapeDetect import ShapeDetect


def show_images(images, kill_later=True):
    """This function show the images

    Args:
        images (np.ndarray): Image array
        kill_later (bool, optional): To kill or not the window. Defaults to True.
    """
    for index, image in enumerate(images):
        cv2.imshow('Kep', image)
    cv2.waitKey(0)
    if kill_later:
        cv2.destroyAllWindows()

def find_contours(image) -> list:
    """Gets the contours of an image

    Args:
        image (np.ndarray): Image array

    Returns:
        list: contour coordinates
    """
    image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.Canny(blurred, 30, 30)
    #Finding contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def get_rectangles(cnts) -> list:
    """Get rectangels on the image

    Args:
        cnts (list): contours
    Returns:
        (list): Contours of the rectangles
    """
    shape_detector = ShapeDetect()
    # loop over the contours
    possible_boxes = []
    for c in cnts:
        #Get if its a candiadte shape or not
        shape = shape_detector.get_shape(c)
        # then draw the contours and the name of the shape on the image
        if shape == "Rectangle":
            c = c.astype("int")
            possible_boxes.append(c)

    return possible_boxes

def get_answer_boxes(contours):
    """Finds the answer boxes on the image.

    Args:
        contours (list): List of contours on the image
    Returns:
        (list): list of contours
    """
    max_area = -1
    answer_boxes = []
    #Find the 
    for c in contours:
        #Create a rectangle and calculate the area
        x, y, w, h = cv2.boundingRect(c)
        if w * h > max_area:
            max_area = w * h
    #Finding the largest contour boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= max_area * 0.90:
            answer_boxes.append(c)
    return answer_boxes


if '__main__' == __name__:

    image_path ="images/as-1234-1.png"

    ANSWER_KEY = {
        0: 1,
        1: 4,
        2: 0,
        3: 3
    }

    # edge detection
    image = cv2.imread(image_path)
    #Finding the contour
    cnts = find_contours(image) 
    #Getting the answer boxes
    possible_boxes = get_rectangles(cnts)
    answer_boxes = get_answer_boxes(possible_boxes)

    #Drawing the answer boxes
    images = []
    for c in answer_boxes:
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        #images.append(image)
        show_images([image])

    # # edge detection
    # image = cv2.imread(image_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.Canny(blurred, 75, 200)
    # show_images([edged])

    # # find contours in edge detected image
    # cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # docCnt = None

    # allContourImage = image.copy()
    # cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
    # print("Total contours found after edge detection {}".format(len(cnts)))
    # show_images([allContourImage])

    # # finding the document contour
    # if len(cnts) > 0:
    #     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #     for c in cnts:
    #         peri = cv2.arcLength(c, closed=True)
    #         approx = cv2.approxPolyDP(c, epsilon=peri*0.02, closed=True)

    #         if len(approx) == 4:
    #             docCnt = approx
    #             break

    # contourImage = image.copy()
    # cv2.drawContours(contourImage, [docCnt], -1, (0, 0, 255), 2)
    # show_images([contourImage])

    # # Getting the bird's eye view, top-view of the document
    # paper = four_point_transform(image, docCnt.reshape(4, 2))
    # warped = four_point_transform(gray, docCnt.reshape(4, 2))
    # show_images([paper, warped])


    # # Thresholding the document
    # thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # show_images([thresh])

    # # Finding contours in threshold image
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # print("Total contours found after threshold {}".format(len(cnts)))
    # questionCnts = []

    # allContourImage = paper.copy()
    # cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
    # show_images([allContourImage])

    # # Finding the questions contours
    # for c in cnts:
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     ar = w / float(h)

    #     if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
    #         questionCnts.append(c)

    # print("Total questions contours found: {}".format(len(questionCnts)))

    # questionsContourImage = paper.copy()
    # cv2.drawContours(questionsContourImage, questionCnts, -1, (0, 0, 255), 3)
    # show_images([questionsContourImage])

    # # Sorting the contours according to the question
    # questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    # correct = 0
    # questionsContourImage = paper.copy()

    # for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    #     cnts = contours.sort_contours(questionCnts[i: i+5])[0]
    #     cv2.drawContours(questionsContourImage, cnts, -1, (255,0,0), 2)
    #     bubbled = None

    #     for (j, c) in enumerate(cnts):
    #         mask = np.zeros(thresh.shape, dtype="uint8")
    #         cv2.drawContours(mask, [c], -1, 255, -1)

    #         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    #         #show_images([mask])
    #         total = cv2.countNonZero(mask)

    #         if bubbled is None or total > bubbled[0]:
    #             bubbled = (total, j)

    #     color = (0, 0, 255)
    #     k = ANSWER_KEY[q]

    #     if k == bubbled[1]:
    #         color = (0, 255, 0)
    #         correct += 1

    #     cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    # score = (correct / 5.0) * 100
    # print("Score: {:.2f}%".format(score))
    # cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    # show_images([image, paper])
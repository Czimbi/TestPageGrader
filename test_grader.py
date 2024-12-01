from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np
import gradio as gr
import pandas as pd
from ShapeDetect import ShapeDetect

def show_images(image, string, kill_later=True):
  cv2.imshow(string, image)
  cv2.waitKey(0)
  if kill_later:
      cv2.destroyAllWindows()

def edge_detection(image_path):
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(blurred, 50, 150)
  #show_images(edged, "Edged")
  return image,edged,gray

# find contours in edge detected image
def find_contours(edged):
  cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  return cnts

def all_contours(cnts, image):
  allContourImage = image.copy()
  cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
  #print("Total contours found after edge detection {}".format(len(cnts)))
  #show_images(allContourImage,"All contours on the image")
  return allContourImage

# finding the document contour
def find_document_contour(cnts, image):
  if len(cnts) > 0:
      cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

      for c in cnts:
          peri = cv2.arcLength(c, closed=True)
          approx = cv2.approxPolyDP(c, epsilon=peri*0.02, closed=True)

          if len(approx) == 4:
              docCnt = approx
              break

  contourImage = image.copy()
  cv2.drawContours(contourImage, [docCnt], -1, (0, 0, 255), 2)
  #show_images(contourImage, "The document's contour on the image")
  return contourImage,docCnt

# Getting the bird's eye view, top-view of the document
def top_view(image, docCnt, gray):
  paper = four_point_transform(image, docCnt.reshape(4, 2))
  warped = four_point_transform(gray, docCnt.reshape(4, 2))
  #show_images(paper, "The document transformed")
  #show_images(warped, "The grayscaled document transformed")
  return paper,warped

def crop(paper, warped):
  height, width, channels = paper.shape
  # Calculate the height of the top 30%
  top_height = int(0.4 * height)
  right_half_start = width // 2
  paper = paper[top_height:, :right_half_start]

  height, width = warped.shape
  # Calculate the height of the top 30%
  top_height = int(0.4 * height)
  right_half_start = width // 2
  warped = warped[top_height:, :right_half_start]

  #show_images(paper, "The document cropped")
  #show_images(warped, "The grayscaled document cropped")
  return paper,warped

def find_contours_on_desired_area(image_orig) -> list:
    image = image_orig.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #Remove blues colors using HSV
    lower_bound = (100, 30, 10)
    upper_bound = (140, 255, 255)

    # Create mask for the targeted colors
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    result = cv2.inpaint(image, mask, 100, cv2.INPAINT_NS)

    #show_images(result, "The blue color masked from the image")

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
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
        if w * h >= max_area * 0.85:
            answer_boxes.append(c)
    return answer_boxes

# Thresholding the document
def threshold(warped):
  thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_TRIANGLE)[1]
  #show_images(thresh, "The threshed image")
  return thresh

def test_grader_2(questionCnts,thresh,paper, ANSWER_KEY):

  prev_answers = 0
  questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
  correct = 0
  questionsContourImage = paper.copy()
  
  for idx, answer_num in enumerate([4, 2, 4, 2]):

  #contours of the FIRST question
    cnts = questionCnts[prev_answers: prev_answers + answer_num]

    cv2.drawContours(questionsContourImage, cnts, -1, (255,0,0), 2)
    bubbled = None
    bubbled_list = []

    for (j, c) in enumerate(cnts):
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        mask = cv2.bitwise_and(thresh, thresh, mask=mask)

        total_pixels = mask.shape[0] * mask.shape[1]
        black_pixels = total_pixels - cv2.countNonZero(mask)
        if bubbled is None or black_pixels > bubbled[0] + 20:
          bubbled = (black_pixels, j)
        bubbled_list.append(black_pixels)
    color = (0, 0, 255)

    #answer of the FIRST question
    k = ANSWER_KEY[idx]
    empty_check = max(bubbled_list) - min(bubbled_list) >= 20
    # check to see if the bubbled answer is correct
    if k == bubbled[1] and empty_check:
      color = (0, 255, 0)
      correct += 1
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    prev_answers += answer_num

  return correct, paper

def omr_function(image_paths,ANSWER_KEY):
  get_names = lambda x:x.split(".")[0].split("\\")[-1]
  names = list(map(get_names, image_paths))
  scores = []
  for image_path in image_paths:
    image,edged,gray = edge_detection(image_path)
    cnts = find_contours(edged)
    allContourImage = all_contours(cnts, image)
    contourImage, docCnt = find_document_contour(cnts, image)
    paper,warped = top_view(image,docCnt, gray)
    paper,warped = crop(paper, warped)

    image = paper.copy()

    cnts = find_contours_on_desired_area(image)
    possible_boxes = get_rectangles(cnts)
    answer_boxes = get_answer_boxes(possible_boxes)
    for c in answer_boxes:
      cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    #show_images(image, "All the answer boxes found on the desired area")

    thresh = threshold(warped)
    correct,paper = test_grader_2(answer_boxes,thresh,paper, ANSWER_KEY=ANSWER_KEY)
    scores.append((correct / 4.0) * 100)

    
  return pd.DataFrame(list(zip(names, scores)), columns =['Name', 'Score'])

def gradio_omr_function(images, dict_input):
  try:
      input_dict = eval(dict_input)
      if not isinstance(input_dict, dict):
          raise ValueError("Input is not a valid dictionary")
  except Exception as e:
      return f"Invalid dictionary input: {e}"
  
  try:
      result_df = omr_function(images, input_dict)
      return result_df
  except Exception as e:
      return f"Error processing inputs: {e}"

if __name__ == '__main__':

  with gr.Blocks() as demo:
      gr.Markdown("## Test Grader")
      gr.Markdown("Upload test images (not scans) and provide the answer key as a dictionary. Make sure the name of the image is the same as the student's (without any special characters and _ between names eg. instead of Péld Aladár.png use Peld_Aladar.png).")
      
      with gr.Row():
          image_input = gr.File(
              label="Upload Images",
              file_types=[".png", ".jpg", ".jpeg"],
              type="filepath",
              file_count="multiple"
          )
          dict_input = gr.Textbox(
            label="Input Answer Key",
            placeholder='Enter the answer key (indexed from 0) e.g., {0: 1, 1: 1}'
        )
      
      output = gr.Dataframe(label="Output Dataframe", interactive=False)
      
      submit_button = gr.Button("Submit")
      submit_button.click(gradio_omr_function, inputs=[image_input, dict_input], outputs=output)


  demo.launch()

  '''ANSWER_KEY = {
      0: 2,
      1: 1,
      2: 1,
      3: 0
    }'''
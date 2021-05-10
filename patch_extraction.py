# defining important parameters

# These are the default parameters for landmark model

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
# Do not need it since it is not an overlay problem
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

p = "/content/drive/MyDrive/Final Acne Analyzer/Utils/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# defining the main functionality

import cv2
import dlib
import numpy as np
from google.colab.patches import cv2_imshow
import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
import copy
import imageio
import os
import sys
from os import listdir
from os.path import join, isfile, splitext
from scipy import misc

def get_patch(img_path):
  path = img_path
  img = cv2.imread(path)
  p = "/content/drive/MyDrive/Final Acne Analyzer/Utils/shape_predictor_68_face_landmarks.dat"
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(p)
  gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = detector(gray)

  for face in face:
    x1=face.left()
    y1=face.top()
    x2=face.right()
    y2=face.bottom()

  landmarks=predictor(gray, face)

  width_ratio = 1.5 # controls the width of the forehead if One Eye model is used to infer the forehead
  top_ratio = 1.5         # controls the height of the forehead if One Eye model is used to infer the forehead
  down_ratio = 4.5        # controls how much we are going down from the detected eye's by the One Eye model
  cheek_width_ratio = 2.8 # controls the width of the cheek skin patch if One Eye model is used
  forehead_ratio = 0.3    # controls the height of forehead when facial landmark model is working


  img_height, img_width = img.shape[0:2] #get the image height and width. Image data is in the format of [height, width, channel]
  min_dim = min(img_height, img_width)
  min_face_size = min_dim * 0.2 # Specify the minimal face size. Heuristic.
  min_eye = min_face_size * 0.2 # specify the minimal eye size.
  min_eye_area = min_eye ** 2 # specify the miniaml area of the eye. This is used screen detected eyes by the OneEye model.
                              # Keep in mind, the One Eye model will identify whatever looks like an eye. We need to screen out
                              # those that are too small.
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img, landmarks = read_im_and_landmarks(path)
  imageName = splitext(os.path.basename(path))[0]

  mask = get_face_mask(img, landmarks)
  face_x_min = int(max(0, np.asarray(min(landmarks[:,0])).flatten()[0])) #Get the minimal value of the detected landmarks in x
  face_x_max = int(min(img_width, np.asarray(max(landmarks[:,0])).flatten()[0])) # Get the maximal value of the detected landmarks in x
  face_y_min = int(max(0, np.asarray(min(landmarks[:,1])).flatten()[0])) # Get the minimal value of the detected landmarks in y
  face_y_max = int(min(img_height, np.asarray(max(landmarks[:,1])).flatten()[0])) # Get the maximal value of the detected landmarks in y
  face_height = face_y_max - face_y_min # Get the height of face
  forehead_height = int(face_height * forehead_ratio) # Ideally, forehead height should be 1/2 of the height between eyebrow and bottom of chin
                                                      # We choose forehead_ratio = 0.3 to avoid hairs on the forehead.
  new_face_y_min = max(0, face_y_min - forehead_height) # new_face_y_min is the top edge of the forehead.
  right_brow_landmarks = landmarks[RIGHT_BROW_POINTS,:]
  left_brow_landmarks = landmarks[LEFT_BROW_POINTS,:]
  right_eye_landmarks = landmarks[RIGHT_EYE_POINTS,:]
  left_eye_landmarks = landmarks[LEFT_EYE_POINTS,:]
  mouse_landmarks = landmarks[MOUTH_POINTS,:]
  ########################
  # Get the forehead patch
  ########################
  [right_brow_min_x, left_brow_max_x] = \
      [max(0, np.min(np.array(right_brow_landmarks[:,0]))), min(img_width, np.max(np.array(left_brow_landmarks[:,0])))]
  brow_min_y = min(np.min(np.array(right_brow_landmarks[:,1])),np.min(np.array(left_brow_landmarks[:,1])))
  forehead_x_min = right_brow_min_x # forehead starts at the left landmark of the right eye brow
  forehead_x_max = left_brow_max_x
  forehead_y_min = max(0, brow_min_y - forehead_height)
  forehead_y_max = min(brow_min_y, forehead_y_min + forehead_height)
  forehead_region = img[forehead_y_min:forehead_y_max, forehead_x_min:forehead_x_max, :]
  # BGR image needs to be converted to RGB before saving as image file
  forehead_region = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2RGB)
  #imageio.imwrite(forehead_file_name, forehead_region)

  chin_x_min = np.max(np.array(right_eye_landmarks[:,0])) #In x direction, chin patch will be between the two most inner
                                                          #points of eyebrows
  chin_x_max = np.min(np.array(left_eye_landmarks[:,0]))
  chin_y_min = np.max(np.array(mouse_landmarks[:,1])) #In y direction, chin patch starts at the lowest point of mouse landmarks
  chin_y_max = face_y_max # In y direction, chin patch ends at the lowest point of face
  chin_region = img[chin_y_min:chin_y_max, chin_x_min:chin_x_max, :]
  chin_region = cv2.cvtColor(chin_region, cv2.COLOR_BGR2RGB)
  #imageio.imwrite(chin_file_name, chin_region)

  # points for right cheek

  gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = detector(gray)

  for face in face:
    x1=face.left()
    y1=face.top()
    x2=face.right()
    y2=face.bottom()

  landmarks=predictor(gray, face)

  x0, x1, x2, x3, x4, x5, x6 = landmarks.part(0).x, landmarks.part(1).x, landmarks.part(2).x, landmarks.part(3).x,  landmarks.part(4).x,  landmarks.part(5).x, landmarks.part(6).x
  y0, y1, y2, y3, y4, y5, y6 = landmarks.part(0).y, landmarks.part(1).y, landmarks.part(2).y, landmarks.part(3).y,  landmarks.part(4).y,  landmarks.part(5).y, landmarks.part(6).y

  l1, l2 = landmarks.part(40).x, landmarks.part(40).y

  pts = np.array([[x0, y0], [x1, y1], [x2, y2],[x3, y3], [x4, y4],[x5, y5], [x6, y6], [l1, l2] ], np.int32)

  masked_image =  mask_image(img, pts)

  right_cheek = remove_bg(masked_image)

  #points for the left cheek
  x0, x1, x2, x3, x4, x5, x6 = landmarks.part(10).x, landmarks.part(11).x, landmarks.part(12).x, landmarks.part(13).x,  landmarks.part(14).x,  landmarks.part(15).x, landmarks.part(16).x
  y0, y1, y2, y3, y4, y5, y6 = landmarks.part(10).y, landmarks.part(11).y, landmarks.part(12).y, landmarks.part(13).y,  landmarks.part(14).y,  landmarks.part(15).y, landmarks.part(16).y

  l1, l2 = landmarks.part(46).x, landmarks.part(46).y

  pts = np.array([[x0, y0], [x1, y1], [x2, y2],[x3, y3], [x4, y4],[x5, y5], [x6, y6], [l1, l2] ], np.int32)

  masked_image =  mask_image(img, pts)

  left_cheek = remove_bg(masked_image)

  imageio.imwrite('Left_Cheek.png', left_cheek)

  return forehead_region, chin_region, right_cheek, left_cheek

# defining useful methods

# a method that takes the image, applies the mask on the given facial landmark points
def mask_image(image, pts):
    mask = np.ones(image.shape, dtype=np.uint8)
    mask.fill(255)
    cv2.fillPoly(mask, [pts], 0)
    masked_image = cv2.bitwise_or(image, mask)
    return masked_image


def remove_bg(masked_image):
    # removing the white background
    # load image
    img = masked_image
    # convert to graky
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold input image as mask
    mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    # negate mask
    mask = 255 - mask
    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # anti-alias the mask -- blur then stretch
    # blur alpha channel
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    # linear stretch so that 127.5 goes to 0, but 255 stays 255
    mask = (2 * (mask.astype(np.float32)) - 255.0).clip(0, 255).astype(np.uint8)
    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)
    im = np.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])
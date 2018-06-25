
##
## This is used to estimate the top of head for Kwon or Vankayalapati
## Stefan Simmer 2017/2018


## how to use ##
## Get Face detected image from DLIB/OPENCV stuff and then increase the frame by xy so that the face is still within the image
## then perform edge detection and go top down until the first edge is hit - save this information and add the delta from top (y)
## to the initial frame position

 ## ### Ellipsis Detection  http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html

import cv2 # OpenCV2
import sys 
import os 
from os import listdir
from os.path import isfile, join
import dlib

import numpy as np
from matplotlib import pyplot as plt
import math
import scipy as sp

# Where to take the images From
input_folder_CG = os.path.join('imageSets/CalTech/trunc')

IMAGE_SIZE_X = 350
IMAGE_SIZE_Y = 350
TOP_BORDER_CG = 14
CANNY_A = 100	
CANNY_B = 100

# Create CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# Create Cascade
cascade_path = os.path.join("dlibStuff/haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascade_path)
predictor_path= os.path.join("dlibstuff/","shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(predictor_path)  

################ SECTION HELPER ###################################################################
# Returns a List of files in a given Folder
def GetFilesFromFolder(folder):
	return [f for f in listdir(folder) if isfile(join(folder, f))] # only use subset for testing purpose for now

# gets an Image, performs Canny Edge Detection and returns the result Image
def PerformEdgeDetection(image):
	return cv2.Canny(image,CANNY_A,CANNY_B)

## Calculate prolongation of nose ray to image border y = 0 -> returns the new x value at y = 0
def CalculateEndpoint(x_start,y_start,noseangle):
	res = y_start*np.tan(noseangle*np.pi/180)
	return x_start + res

## calculate the nose angle - which is also an indicator of head rotation
def CalculateNoseAngle(shape):
	xlist = []
	ylist = []

	for i in range(0,68):
		xlist.append(shape.item(i,0))
		ylist.append(shape.item(i,1))

	xmean = np.mean(xlist)
	ymean = np.mean(ylist)

	xcentral = [(x-xmean) for x in xlist] #Calculate distance centre <-> other points in both axes
	ycentral = [(y-ymean) for y in ylist]

	noseangle = 0
	#noseangle = int(np.arctan((ylist[27]-ylist[30])/(xlist[30]-xlist[27]))*180/np.pi)
	if xlist[30] == xlist[27]:
		noseangle = 0
	else:
		noseangle = np.arcsin((xlist[30]-xlist[27])/(ylist[27]-ylist[30]))*180/np.pi


	#print("currAngle: {0}".format(noseangle))

	y_eyes = (ylist[38]+ylist[40])/2 - (ylist[44]+ylist[46])/2
	eye_distance = (xlist[44]+xlist[46])/2 - (xlist[38]+xlist[40])/2 
	newAngle = np.arcsin(y_eyes/eye_distance)*180/np.pi
	#print("eye_y {0}".format(y_eyes))
	#print("distance: {0}".format(eye_distance))
	#print("new_angle: {0}".format(newAngle))

	return -newAngle

def interpolate_pixels_along_line(x0, y0, x1, y1):
    """Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along a
    straight line, given two points (x0, y0) and (x1, y1)

    Wikipedia article containing pseudo code that function was based off of:
        http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm
    """
    pixels = []
    steep = abs(y1 - y0) > abs(x1 - x0)

    # Ensure that the path to be interpolated is shallow and from left to right
    if steep:
        t = x0
        x0 = y0
        y0 = t

        t = x1
        x1 = y1
        y1 = t

    if x0 > x1:
        t = x0
        x0 = x1
        x1 = t

        t = y0
        y0 = y1
        y1 = t

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx  # slope

    # Get the first given coordinate and add it to the return list
    x_end = round(x0)
    y_end = y0 + (gradient * (x_end - x0))
    xpxl0 = x_end
    ypxl0 = round(y_end)
    if steep:
        pixels.extend([(ypxl0, xpxl0), (ypxl0 + 1, xpxl0)])
    else:
        pixels.extend([(xpxl0, ypxl0), (xpxl0, ypxl0 + 1)])

    interpolated_y = y_end + gradient

    # Get the second given coordinate to give the main loop a range
    x_end = round(x1)
    y_end = y1 + (gradient * (x_end - x1))
    xpxl1 = x_end
    ypxl1 = round(y_end)

    # Loop between the first x coordinate and the second x coordinate, interpolating the y coordinates
    for x in range(xpxl0 + 1, xpxl1):
        if steep:
            pixels.extend([(math.floor(interpolated_y), x), (math.floor(interpolated_y) + 1, x)])

        else:
            pixels.extend([(x, math.floor(interpolated_y)), (x, math.floor(interpolated_y) + 1)])

        interpolated_y += gradient

    # Add the second given coordinate to the given list
    if steep:
        pixels.extend([(ypxl1, xpxl1), (ypxl1 + 1, xpxl1)])
    else:
        pixels.extend([(xpxl1, ypxl1), (xpxl1, ypxl1 + 1)])

    return pixels

# give landmarks and unprocessed image (!) - gets ToH ToupleList
def GetTopOfHeadValues(landmarks,image):

	_getDistance = lambda a,b: b - a if b > a else a - b 
	# Berechnung der Verschiebung von X durch Winkel noseangle
	_CalculateEndpoint = lambda x_start, y_start, noseangle: x_start + y_start*np.tan(noseangle*np.pi/180)
	# make sure it doesn't get out of image border
	_f = lambda x: int(x) if x > 0 else 0

	noseangle = CalculateNoseAngle(landmarks)

	StartX = round((landmarks.item(22,0)+landmarks.item(21,0))/2)
	StartY = round((landmarks.item(22,1)+landmarks.item(21,1))/2)
	endPoint = round(_CalculateEndpoint(StartX,StartY,noseangle))
	# 12 is border from top - don't count until there, image set CGNET is broken
	pixels = interpolate_pixels_along_line(StartX,StartY,int(endPoint),TOP_BORDER_CG)

	# result = [p for p in pixels if edgesImageCopy[p[1],p[0]] == 255] !!!!!!!!!!!!!!!!!!!!!!
	result = [p for p in pixels if image[p[1],p[0]] == 255]

	topOfHeadCoords = tuple([ int(sum(y) / len(y)) for y in zip(*result)])
	
	# Distance 8<->27 * 0.8 as minimum value DO NOT USE UNLESS ABSOLUTELY NECCESSARY; CORRELATION BETWEEN FEATURES
	ToH_YMin = (int(_CalculateEndpoint(StartX,StartY,noseangle)), _f(((StartY-_getDistance(landmarks.item(8,1),landmarks.item(27,1))*0.85))))
	ToH_YMax = (int(_CalculateEndpoint(StartX,StartY,noseangle)), _f(((StartY-_getDistance(landmarks.item(8,1),landmarks.item(27,1))*1.05))))
	# Min Y value in Touple Result set ( closest to top window border) is chosen as Max - can be lower than min actually
	ToH_YMaxEdge = (int(_CalculateEndpoint(StartX,StartY,noseangle)), min(result, default=(0,0), key=lambda  item:item[1] )[1])
	#print(ToH_YMax, ToH_YMaxEdge)

	def average(xs):
		N = float(len(xs))
		return tuple(int(sum(col)/N) for col in zip(*xs))
	
	# panic
	if topOfHeadCoords == () :
		topOfHeadCoords = average([ToH_YMin,ToH_YMax])
	#has to be lower than max edge
	# bad edge detection, should be lower than min in all cases
	if (ToH_YMaxEdge[1] > ToH_YMin[1]):
		ToH_YMaxEdge = ToH_YMin
	ToH_final = average([ToH_YMaxEdge,ToH_YMin,topOfHeadCoords])

	if ToH_final[1] < ToH_YMaxEdge[1]:
		ToH_final = ToH_YMaxEdge

	return topOfHeadCoords,ToH_YMin,ToH_YMax,ToH_YMaxEdge,ToH_final

################ SECTION Image Processing #########################################################

def FindFacesInImage(image):
	#convert to gray
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# find faces
	faces = faceCascade.detectMultiScale(image, scaleFactor = 1.2 , minNeighbors = 4, minSize = (100, 100), flags = cv2.CASCADE_SCALE_IMAGE)
	#faces = faceCascade.detectMultiScale(image, scaleFactor=2, minNeighbors=4, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
	# for all found faces ( should be 1 )
	for (x,y,w,h) in faces:
		dlib_rect = dlib.rectangle( np.long(x),np.long(y),np.long(x+w),np.long(y+h))

		detected_landmarks = predictor(image, dlib_rect).parts()
		landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])
		# y+h nach unten
		# x+w nach rechts vermutlich
		image = image[y:y+h,x:x+w]
		# !!!! 
		out = cv2.resize(image,(IMAGE_SIZE_X,IMAGE_SIZE_Y))
		return  out,dlib_rect,landmarks

def Do():
	for file in GetFilesFromFolder(input_folder_CG):
		img = cv2.imread(os.path.join(input_folder_CG,file),0)

		show68Image = img.copy()
		showRectImage = img.copy()

		#Step 1 & 2: Face Detection and Landmark Extraction
		faceDetectionImage,faceRectangle,landmarks = FindFacesInImage(img)
		noseAngle = CalculateNoseAngle(landmarks)
		

		#Step 3: Edge Detection
		edgesImage = PerformEdgeDetection(img)
		edgesImageCopy = edgesImage.copy() # copy grayscale version for edge analysis 

		#STEP 3.5 Make image Color again for viewing purpose
		edgesImage = cv2.cvtColor(edgesImage,cv2.COLOR_GRAY2RGB)

		#Step4 Show all interesting Points for Kwon or Vank in Image
		edgesImage = cv2.rectangle(edgesImage,(faceRectangle.left(),faceRectangle.top()),(faceRectangle.right(),faceRectangle.bottom()), color = (255, 0, 0))
		showRectImage = cv2.rectangle(showRectImage,(faceRectangle.left(),faceRectangle.top()),(faceRectangle.right(),faceRectangle.bottom()), color = (255, 0, 0))

		StartX = round((landmarks.item(22,0)+landmarks.item(21,0))/2)
		StartY = round((landmarks.item(22,1)+landmarks.item(21,1))/2)
		endPoint = round(CalculateEndpoint(StartX,StartY,noseAngle))

		# DONT DO IMAGE PROCESSING PAST HERE - IMAGE HAS DRAWINGS INSIDE
		cv2.putText(edgesImage, "C", (StartX,StartY),  fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color = (255, 0, 255))
		cv2.circle(edgesImage, (StartX,StartY), 3, color = (255, 0, 255))  # draw points on the landmark positions
	
		## Render NoseAngleLine into Image
		cv2.line(edgesImage,(StartX,StartY),(int(endPoint),0),(255,0,255), 3)
	
		topOfHeadCoords, ToH_YMin, ToH_YMax, ToH_YMaxEdge,ToH_final = GetTopOfHeadValues(landmarks,edgesImageCopy) # get ToH values from function


		# Render ToH values into image
		cv2.putText(edgesImage, "ToH", topOfHeadCoords, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color=(0, 0, 255))
		cv2.circle(edgesImage, topOfHeadCoords, 3, color = (0, 255, 255)) 

		cv2.putText(edgesImage, "ToH_Min", ToH_YMin, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color=(0, 0, 255))
		cv2.circle(edgesImage, ToH_YMin, 3, color = (0, 255, 255)) 

		cv2.putText(edgesImage, "ToH_Max", ToH_YMax, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color=(0, 0, 255))
		cv2.circle(edgesImage, ToH_YMax, 3, color = (0, 255, 255)) 

		cv2.putText(edgesImage, "ToH_Max_Edge", ToH_YMaxEdge, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color=(0, 0, 255))
		cv2.circle(edgesImage, ToH_YMaxEdge, 3, color = (0, 255, 255)) 

		cv2.putText(edgesImage, "ToH_final", ToH_final, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color=(0, 255, 255))
		cv2.circle(edgesImage, ToH_final, 3, color = (128, 128, 128)) 

		## Render Landmarks into Image
		for idx, point in enumerate(landmarks):
			pos = (point[0, 0], point[0, 1])
			# annotate the positions
			cv2.putText(edgesImage, str(idx), pos,  fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color = (0, 0, 255))
			cv2.circle(edgesImage, pos, 3, color = (0, 255, 255))  # draw points on the landmark positions
			cv2.putText(show68Image, str(idx), pos,  fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.4,  color = (0, 0, 255))
			cv2.circle(show68Image, pos, 3, color = (0, 255, 255))  # draw points on the landmark positions

		#Step5 Write all these Results into CSV file with Annotation in Mind
	
		#STEP 4.5 Make image Color again for viewing purpose

		#plt.subplot(121),plt.imshow(img,cmap = 'gray')
		#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
		#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
		#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
		#plt.show()

		window = cv2.imshow("test",edgesImage)
		#window = cv2.imshow("test",show68Image)
		cv2.waitKey(0);
		cv2.destroyAllWindows()

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   Do()



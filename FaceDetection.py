
## Find Faces in Images and Export to Image File ##
## Stefan Simmer 
## June 2018
## Master Thesis

import cv2
import dlib
import math
import numpy as np
import itertools
import os
import sys

from os import listdir
from os.path import isfile, join

IMAGE_SIZE_X = 350
IMAGE_SIZE_Y = 350

# Define Image Input and Output Folders
input_folder_CG = os.path.join('imageSets/CGPlus/trunc')
output_folder_CG = os.path.join('imageSets/CGPlus/trunc/out')

input_folder_FG = os.path.join('imageSets/FGNet/trunc')
output_folder_FG = os.path.join('imageSets/FGNet/trunc/out')

#input_folder_CG = os.path.join('imageSets/CalTech/trunc')
#output_folder_CG = os.path.join('imageSets/CalTech/trunc/out')
#input_folder_CG = os.path.join('imageSets/FEI/trunc')
#output_folder_CG = os.path.join('imageSets/FEI/trunc/out')

# Create CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# Create Cascade
cascade_path = os.path.join("dlibStuff/haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(cascade_path)

# process one image
def FindFacesInImage(image):
	#conver to gray
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# find faces
	faces = faceCascade.detectMultiScale(image, scaleFactor=1.01, minNeighbors=4, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
	#faces = faceCascade.detectMultiScale(image, scaleFactor=2, minNeighbors=4, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
	# for all found faces ( should be 1 )
	for (x,y,w,h) in faces:
		dlib_rect = dlib.rectangle( np.long(x),np.long(y),np.long(x+w),np.long(y+h))
		print(dlib_rect)
		# y+h nach unten
		# x+w nach rechts vermutlich
		gray = gray[y:y+h,x:x+w]
		out = cv2.resize(gray,(IMAGE_SIZE_X,IMAGE_SIZE_Y))
		return  out

# Use OpenCV2 to read an Image from File in Folder and Return it
def GetCV2Image(folder, file):
	image = cv2.imread(os.path.join(folder,file))
	return image

# Returns a List of files in a given Folder
def GetFilesFromFolder(folder):
	return [f for f in listdir(folder) if isfile(join(folder, f))] # only use subset for testing purpose for now

# Process all images in CG and write to CG->out
for file in GetFilesFromFolder(input_folder_CG):
	image = GetCV2Image(input_folder_CG,file)
	print("Processing: {0}/{1}".format(input_folder_CG,file))
	writeImage = FindFacesInImage(image)
	if writeImage is not None:
		cv2.imwrite(output_folder_CG+"\\"+file,writeImage)
 # Process all images in FG and write to FG->out	
for file in GetFilesFromFolder(input_folder_FG):
	image = GetCV2Image(input_folder_FG,file)
	print("Processing: {0}/{1}".format(input_folder_CG,file))
	writeImage = FindFacesInImage(image)
	if writeImage is not None:
		cv2.imwrite(output_folder_FG+"\\"+file,writeImage)




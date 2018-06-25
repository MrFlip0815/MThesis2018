import cv2
import pandas as pd
import numpy as np
import argparse
import os
import sys
from os.path import isfile, join
from os import listdir
import time
import csv

# Mouse Events in CV2 https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

image_folder = os.path.join('imageSets/FEI/trunc_ratio')
outFile = os.path.join(image_folder,'OUT.MANUAL.{0}.csv'.format(time.time()))

# Returns a List of files in a given Folder
def GetFilesFromFolder(folder):
	return [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith(".jpg")] # only use subset for testing purpose for now
   
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
clicked = False

def userClickedEvent(event, x, y, flags, param):
	# grab references to the global variables
	global refPt

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is beingas
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		print(refPt)
		clicked=True

files = GetFilesFromFolder(image_folder)

final_list = []

def DO():
	for f in files:
		clicked=False
		print(f)
		image = cv2.imread(os.path.join(image_folder,f))
	
		while not clicked:
			window = cv2.imshow("image",image)
			cv2.setMouseCallback("image", userClickedEvent)
			cv2.waitKey(0);
			clicked=True

			tmp = []
			tmp.extend([f])
			tmp.extend(refPt[0])
			final_list.append(tmp)
			print(final_list)
		cv2.destroyAllWindows()

DO()

with open(outFile, 'w') as myFile:
	wr = csv.writer(myFile)
	wr.writerows(final_list)
	  
	

#
#with open(os.path.join(image_folder_FEI,'pythonthings.txt'), 'r') as myFile:
#	reader = csv.reader(myFile)
#	your_list = list(reader)
#	print(your_list)




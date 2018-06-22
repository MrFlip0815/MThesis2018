
# landmark stuff

import numpy as np
import json
import random
import itertools 
import shutil # shell utils
import os
import re # regular expressions 
import sys # 
import cv2
import dlib
from os import listdir
from os.path import isfile, join
import csv


import Headstimation as head

# use trunc pictures - they are still in same aspect ratio
image_folder_CK = os.path.join('imageSets/CGPlus/trunc')
image_folder_FG = os.path.join('imageSets/FGNet/trunc')
image_folder_FEI = os.path.join('imageSets/FEI/trunc')
image_folder_CalTech = os.path.join('imageSets/CalTech/trunc')


# Setup DLIB and OpenCV
cascade_path = os.path.join("dlibstuff/","haarcascade_frontalface_default.xml")
predictor_path= os.path.join("dlibstuff/","shape_predictor_68_face_landmarks.dat")
kwon_ratios = [True,True,True,True,True]
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

faceCascade = cv2.CascadeClassifier(cascade_path)  
# create the landmark predictor  
predictor = dlib.shape_predictor(predictor_path)  

def GetFilesFromFolder(folder):
	return [f for f in listdir(folder) if isfile(join(folder, f))] # only use subset for testing purpose for now

# these are the image files 
files_CK = GetFilesFromFolder(image_folder_CK)
files_FG = GetFilesFromFolder(image_folder_FG)
files_FEI = GetFilesFromFolder(image_folder_FEI)
files_CalTech = GetFilesFromFolder(image_folder_CalTech)
# we also need the annotation files 
annotation_CK = os.path.join('imageSets/CGPlus','annotation.csv')
annotation_FG = os.path.join('imageSets/FGNet','annotation.csv')
annotation_FEI = os.path.join('imageSets/FEI','FEIAnnotation.csv')
annotation_CalTech = os.path.join('imageSets/CalTech','CalTechAnnotation.csv') # TODO TODO TODO

# define the output CSV files

CK_KWON_female_csv = os.path.join('imageSets/CGPlus','CK.KWON.female.csv')
CK_KWON_male_csv = os.path.join('imageSets/CGPlus','CK.KWON.male.csv')

FG_KWON_female_csv = os.path.join('imageSets/FGNet','FG.KWON.female.csv')
FG_KWON_male_csv = os.path.join('imageSets/FGNet','FG.KWON.male.csv')

FEI_KWON_female_csv = os.path.join('imageSets/FEI','FEI.KWON.female.csv')
FEI_KWON_male_csv = os.path.join('imageSets/FEI','FEI.KWON.male.csv')

CalTech_KWON_female_csv = os.path.join('imageSets/CalTech','CalTech.KWON.female.csv')
CalTech_KWON_male_csv = os.path.join('imageSets/CalTech','CalTech.KWON.male.csv')


# load Annotation CSV FG
annotation_data_FG = []
with open (annotation_FG,'r') as csvReader:
	spamreader = csv.reader(csvReader,delimiter=',')
	for row in spamreader:
		annotation_data_FG.append(row)

# load Annotation CSV FEI
annotation_data_FEI = []
with open (annotation_FEI,'r') as csvReader:
	spamreader = csv.reader(csvReader,delimiter=',')
	for row in spamreader:
		if row != []:
			annotation_data_FEI.append(row)
# load Annotation CSV CK
annotation_data_CK = []
with open (annotation_CK,'r') as csvReader:
	spamreader = csv.reader(csvReader,delimiter=',')
	for row in spamreader:
		annotation_data_CK.append(row)

# TODO CalTech
annotation_data_CalTech = []
with open (annotation_CalTech,'r') as csvReader:
	spamreader = csv.reader(csvReader,delimiter=',')
	for row in spamreader:
		annotation_data_CalTech.append(row)
##



def get_simple_landmarks(image):

	faces = faceCascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

	if len(faces) == 0:
		return []

	

	for (x,y,w,h) in faces:
		dlib_rect = dlib.rectangle( np.long(x),np.long(y),np.long(x+w),np.long(y+h))
		detected_landmarks = predictor(image, dlib_rect).parts()
		#print(detected_landmarks)

		points = []
		points = [[p.x, p.y] for p in detected_landmarks]
		# points to array
		landmarks = np.matrix(points)

		# Calculate NoseAngle from landmarks
		noseAngle = head.CalculateNoseAngle(landmarks)
		#print(noseAngle)
		# we need a gray image copy first
		gray_copy = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
		# perform Edge Detection 
		edgesImage = head.PerformEdgeDetection(gray_copy)

		topOfHeadCoords,ToH_YMin,ToH_YMax,ToH_YMaxEdge,ToH_final = head.GetTopOfHeadValues(landmarks,edgesImage)

		# append Toh to array
		points.append([ToH_final[0],ToH_final[1]])

		# create matrix 
		landmarks = np.matrix(points)

		# add new toh to landmark matrix

	#print(landmarks)

	return landmarks # TODOOOOOOOOOOO - return additional parameter as TOP OF HEAD AND REDO EVERYTHING !!!!!!!!!!!!!!!

######################################################
# Windows Version for KWON
######################################################
class KwonRatiosWindows(object):
	def __init__(self, landmarks):
		self.landmarks = landmarks
		self.k_list = [True,True,True,True,True]

	def getPointX(self,n):
		return self.landmarks.item(n,0)
	def getPointY(self,n):
		return self.landmarks.item(n,1)
	def getAvg(self,a,b):
		return (a+b)/2
	def getDistance(self,a,b):
		if(b>a):
			return b-a
		else:
			return a-b
	# average y value of both eyes
	def eyes_y_avg(self):
		eye_left_y = self.getAvg(self.getPointY(40),self.getPointY(38))
		eye_right_y = self.getAvg(self.getPointY(46),self.getPointY(44))
		eye_avg_y = self.getAvg(eye_left_y,eye_right_y)
		return eye_avg_y
	def eye_x_avg(self):
		eye_left_x = self.getAvg(self.getPointX(36),self.getPointX(39))
		eye_right_x = self.getAvg(self.getPointX(45),self.getPointX(42))
		eye_avg_x = self.getAvg(eye_left_x,eye_right_x)
		return eye_avg_x
	# distance between left eye and right eye center 
	def eyeDistance(self):
		eye_left_x = self.getAvg(self.getPointX(36),self.getPointX(39))
		eye_right_x = self.getAvg(self.getPointX(45),self.getPointX(42))
		return self.getDistance(eye_right_x,eye_left_x)
	# distance between y average of eyese and nose bottom
	def eyeNoseDistance(self):
		eyes_y = self.eyes_y_avg()
		nose_y = self.getPointY(33)
		return self.getDistance(eyes_y,nose_y)
	# Y value of mouth center:: TODO SUBJECT TO EVALUATE 60<->62 or 48<->54 or avg of both --- also 61,62,63 upper lip 
	def mouthMiddle_Y(self):
		return self.getAvg(self.getPointY(60),self.getPointY(64))
	def mouthMiddle_X(self):
		return self.getAvg(self.getPointX(60),self.getPointX(64))

	def chinY(self):
		return self.getPointY(8)
	def chinX(self):
		return self.getPointX(8)

	# Y eye to mouth distance
	def eyeMouthDistance(self):
		eyes_y = self.eyes_y_avg()
		mouth_y = self.mouthMiddle_Y()
		return self.getDistance(eyes_y,mouth_y)
	def eyeChinDistance(self):
		eyes_y = self.eyes_y_avg()
		chin = self.chinY()
		return self.getDistance(eyes_y,chin)
	
	# Kwon Ratio 1 
	def kwonRatio1(self):
		return self.eyeDistance()/self.eyeNoseDistance()
	# Kwon Ratio 2 
	def kwonRatio2(self):
		return self.eyeDistance()/self.eyeMouthDistance()
	# Kwon Ratio 3 
	def kwonRatio3(self):
		return self.eyeDistance()/self.eyeChinDistance()
	# Kwon Ratio 4 
	def kwonRatio4(self):
		return self.eyeNoseDistance()/self.eyeMouthDistance()
	# Kwon Ratio 5
	def kwonRatio5(self):
		return self.eyeMouthDistance()/self.eyeChinDistance()
	# returns a JSON document with 5 Kwon Ratios and the landmarks
	def getKwonJson(self):
		return json.dumps({"landmarks":self.landmarks.tolist() ,"ratio1":self.kwonRatio1(),"ratio2":self.kwonRatio2(),"ratio3":self.kwonRatio3(),"ratio4":self.kwonRatio4(),"ratio5":self.kwonRatio5()})

	def getKwonMouth(self):
		return [self.mouthMiddle_X(),self.mouthMiddle_Y()]
	def getKwonLeftEye(self):
		return [self.getAvg(self.getPointX(36),self.getPointX(39)),self.getAvg(self.getPointY(40),self.getPointY(38))]
	def getKwonRightEye(self):
		return [self.getAvg(self.getPointX(45),self.getPointX(42)),self.getAvg(self.getPointY(46),self.getPointY(44))]
	def getKwonChin(self):
		return [self.chinX(),self.chinY()]
	def getKwonNose(self):
		return [self.getPointX(33),self.getPointY(33)]
	def getBetweenEyes(self):
		return [self.eye_x_avg(),self.eyes_y_avg()]
	def kwonALL(self):
		ret_list=[]
		if self.k_list[0] == True:
			ret_list.append(self.kwonRatio1())
		if self.k_list[1] == True:
			ret_list.append(self.kwonRatio2())
		if self.k_list[2] == True:
			ret_list.append(self.kwonRatio3())
		if self.k_list[3] == True:
			ret_list.append(self.kwonRatio4())
		if self.k_list[4] == True:
			ret_list.append(self.kwonRatio5())	
		return ret_list

class KwonRatiosWindowsNEWNEW(object):
	def __init__(self, landmarks):
		self.landmarks = landmarks
		self.k_list = [True,True,True,True,True]

	def getPointX(self,n):
		return self.landmarks.item(n,0)
	def getPointY(self,n):
		return self.landmarks.item(n,1)
	def getAvg(self,a,b):
		return (a+b)/2
	def getDistance(self,a,b):
		if(b>a):
			return b-a
		else:
			return a-b
	# average y value of both eyes
	def eyes_y_avg(self):
		eye_left_y = self.getAvg(self.getPointY(40),self.getPointY(38))
		eye_right_y = self.getAvg(self.getPointY(46),self.getPointY(44))
		eye_avg_y = self.getAvg(eye_left_y,eye_right_y)
		return eye_avg_y
	def eye_x_avg(self):
		eye_left_x = self.getAvg(self.getPointX(36),self.getPointX(39))
		eye_right_x = self.getAvg(self.getPointX(45),self.getPointX(42))
		eye_avg_x = self.getAvg(eye_left_x,eye_right_x)
		return eye_avg_x
	# distance between left eye and right eye center 
	def eyeDistance(self):
		eye_left_x = self.getAvg(self.getPointX(36),self.getPointX(39))
		eye_right_x = self.getAvg(self.getPointX(45),self.getPointX(42))
		return self.getDistance(eye_right_x,eye_left_x)
	# distance between y average of eyese and nose bottom
	def eyeNoseDistance(self):
		eyes_y = self.eyes_y_avg()
		nose_y = self.getPointY(33)
		return self.getDistance(eyes_y,nose_y)
	# Y value of mouth center:: TODO SUBJECT TO EVALUATE 60<->62 or 48<->54 or avg of both --- also 61,62,63 upper lip 
	def mouthMiddle_Y(self):
		return self.getAvg(self.getPointY(60),self.getPointY(64))
	def mouthMiddle_X(self):
		return self.getAvg(self.getPointX(60),self.getPointX(64))

	def chinY(self):
		return self.getPointY(8)
	def chinX(self):
		return self.getPointX(8)

	def HeadX(self):
		return self.getPointX(68) # maybe 68
	def HeadY(self):
		return self.getPointY(68) # maybe 68

	def eyeHeadDistance(self):
		eyes_y = self.eyes_y_avg()
		head_y = self.HeadY()
		return self.getDistance(eyes_y,head_y)


	# Y eye to mouth distance
	def eyeMouthDistance(self):
		eyes_y = self.eyes_y_avg()
		mouth_y = self.mouthMiddle_Y()
		return self.getDistance(eyes_y,mouth_y)
	def eyeChinDistance(self):
		eyes_y = self.eyes_y_avg()
		chin = self.chinY()
		return self.getDistance(eyes_y,chin)
	
	def chinHeadDistance(self):
		chin_y = self.chinY()
		head_y = self.HeadY()
		return self.getDistance(chin_y,head_y)

	# Kwon Ratio 1 
	def kwonRatio1(self):
		return self.eyeDistance()/self.eyeNoseDistance()
	# Kwon Ratio 2 
	def kwonRatio2(self):
		return self.eyeDistance()/self.eyeHeadDistance()
	# Kwon Ratio 3 
	def kwonRatio3(self):
		return self.eyeMouthDistance()/self.eyeHeadDistance()
	# Kwon Ratio 4 
	def kwonRatio4(self):
		return self.eyeMouthDistance()/self.eyeMouthDistance()
	# Kwon Ratio 5
	def kwonRatio5(self):
		return self.eyeHeadDistance()/self.chinHeadDistance()


	# returns a JSON document with 5 Kwon Ratios and the landmarks
	def getKwonJson(self):
		return json.dumps({"landmarks":self.landmarks.tolist() ,"ratio1":self.kwonRatio1(),"ratio2":self.kwonRatio2(),"ratio3":self.kwonRatio3(),"ratio4":self.kwonRatio4(),"ratio5":self.kwonRatio5()})

	def getKwonMouth(self):
		return [self.mouthMiddle_X(),self.mouthMiddle_Y()]
	def getKwonLeftEye(self):
		return [self.getAvg(self.getPointX(36),self.getPointX(39)),self.getAvg(self.getPointY(40),self.getPointY(38))]
	def getKwonRightEye(self):
		return [self.getAvg(self.getPointX(45),self.getPointX(42)),self.getAvg(self.getPointY(46),self.getPointY(44))]
	def getKwonChin(self):
		return [self.chinX(),self.chinY()]
	def getKwonNose(self):
		return [self.getPointX(33),self.getPointY(33)]
	def getBetweenEyes(self):
		return [self.eye_x_avg(),self.eyes_y_avg()]
	def getKwonHead(self):
		return [self.HeadX(),self.HeadY()]

	def kwonALL(self):
		ret_list=[]
		if self.k_list[0] == True:
			ret_list.append(self.kwonRatio1())
		if self.k_list[1] == True:
			ret_list.append(self.kwonRatio2())
		if self.k_list[2] == True:
			ret_list.append(self.kwonRatio3())
		if self.k_list[3] == True:
			ret_list.append(self.kwonRatio4())
		if self.k_list[4] == True:
			ret_list.append(self.kwonRatio5())	
		return ret_list

############### Get Gender from Annotation File for Filename ##########################
def getGenderForSubstringFG(substring):
	for r in annotation_data_FG:
		if r[1][0:3] == substring:
			return r[2]

def getGenderForSubstringFEI(substring):
	for r in annotation_data_FEI:
		if r[0] == substring:
			return r[1]

def getGenderForSubstringCK(substring):
	for r in annotation_data_CK:
		if r[0] == substring:
			return r[1] 

def getGenderForSubstringCalTech(substring):
	print(substring)
	for r in annotation_data_CalTech:
		print(r)
		if r[0] == substring:
			return r[1] 

#Use OpenCV2 to read an Image from File in Folder and return it
def GetCV2Image(folder, file):
	image = cv2.imread(os.path.join(folder,file))
	return image

######

kwon_CK_male = []
kwon_CK_female = []
kwon_FG_male = []
kwon_FG_female = []
kwon_FEI_male = []
kwon_FEI_female = []
kwon_CalTech_female = []
kwon_CalTech_male = []

#for f in files_CK:
#	img = GetCV2Image(image_folder_CK,f)
#	marks = get_simple_landmarks(img)
#	gender = getGenderForSubstringCK((f.split('_'))[0])
#	kwon_data = KwonRatiosWindows(marks).kwonALL()

#	if gender == 'male':
#		tmp = []
#		tmp.extend([f,gender])
#		tmp.extend(kwon_data)
#		kwon_CK_male.append(tmp)
#	else:
#		tmp = []
#		tmp.extend([f,gender])
#		tmp.extend(kwon_data)
#		kwon_CK_female.append(tmp)
#	print("CK+ File: {0} {1}".format(f,gender))

for f in files_FG:
	img = GetCV2Image(image_folder_FG,f)
	marks = get_simple_landmarks(img)
	if marks == []: # skip images with no landmarks
		continue
	gender = getGenderForSubstringFG((f.split('.'))[0][0:3])
	kwon_data = KwonRatiosWindows(marks).kwonALL()
	if gender == 'male':
		tmp = []
		tmp.extend([f,gender])
		tmp.extend(kwon_data)
		kwon_FG_male.append(tmp)
	else:
		tmp = []
		tmp.extend([f,gender])
		tmp.extend(kwon_data)
		kwon_FG_female.append(tmp)
	print("FG File: {0} {1}".format(f,gender))

#for f in files_FEI:
#	img = GetCV2Image(image_folder_FEI,f)
#	marks = get_simple_landmarks(img)
#	if marks == []: # skip images with no landmarks
#		continue
#	gender = getGenderForSubstringFEI((f.split('.'))[0])
#	kwon_data = KwonRatiosWindows(marks).kwonALL()
#	if gender == 'male':
#		tmp = []
#		tmp.extend([f,gender])
#		tmp.extend(kwon_data)
#		kwon_FEI_male.append(tmp)
#	else:
#		tmp = []
#		tmp.extend([f,gender])
#		tmp.extend(kwon_data)
#		kwon_FEI_female.append(tmp)
#	print("FEI File: {0} {1}".format(f,gender))

#for f in files_CalTech:
#	pass
#	img = GetCV2Image(image_folder_CalTech,f)
#	marks = get_simple_landmarks(img)
#	if marks == []: # skip images with no landmarks
#		continue
#	gender = getGenderForSubstringCalTech((f.split('.'))[0])
#	print(f)
#	print(gender)
#	kwon_data = KwonRatiosWindowsNEWNEW(marks).kwonALL()
#	if gender == 'male':
#		tmp = []
#		tmp.extend([f,gender])
#		tmp.extend(kwon_data)
#		kwon_CalTech_male.append(tmp)
#	else:
#		tmp = []
#		tmp.extend([f,gender])
#		tmp.extend(kwon_data)
#		kwon_CalTech_female.append(tmp)
#	print("CakTech File: {0} {1}".format(f,gender))

# CK Female
#with open(CK_KWON_female_csv, 'w', newline='') as csvFile:
#    wr = csv.writer(csvFile)
#    wr.writerows(kwon_CK_female)
## CK Male
#with open(CK_KWON_male_csv, 'w', newline='') as csvFile:
#    wr = csv.writer(csvFile)
#    wr.writerows(kwon_CK_male)
# FG Female
with open(FG_KWON_female_csv, 'w', newline='') as csvFile:
    wr = csv.writer(csvFile)
    wr.writerows(kwon_FG_female)
# FG male
with open(FG_KWON_male_csv, 'w', newline='') as csvFile:
    wr = csv.writer(csvFile)
    wr.writerows(kwon_FG_male)
## FEI female
#with open(FEI_KWON_female_csv, 'w', newline='') as csvFile:
#    wr = csv.writer(csvFile)
#    wr.writerows(kwon_FEI_female)
#with open(FEI_KWON_male_csv, 'w', newline='') as csvFile:
#    wr = csv.writer(csvFile)
#    wr.writerows(kwon_FEI_male)

#with open(CalTech_KWON_female_csv, 'w', newline='') as csvFile:
#    wr = csv.writer(csvFile)
#    wr.writerows(kwon_CalTech_female)

#with open(CalTech_KWON_male_csv, 'w', newline='') as csvFile:
#    wr = csv.writer(csvFile)
#    wr.writerows(kwon_CalTech_male)

#write all the data to CSV in their corresponding folder




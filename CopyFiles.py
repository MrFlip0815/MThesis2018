
## Copy Images from CG/trunc to male/female depending on csv gender

import csv
import os
import shutil

from os import listdir
from os.path import isfile, join

# CK+
def doCK():
	# Define Image Input and Output Folders
	csv_file = os.path.join('imageSets/CGPlus','annotation.csv')
	output_folder_CG = os.path.join('imageSets/CGPlus/trunc/out')
	train_folder = os.path.join('imageSets/CGPlus/train')
	validate_folder = os.path.join('imageSets/CGPlus/validate')

	result = []
	with open (csv_file,'r') as csvReader:
		spamreader = csv.reader(csvReader,delimiter=',')
		for row in spamreader:
			result.append(row)


	def GetFilesFromFolder(folder):
		return [f for f in listdir(folder) if isfile(join(folder, f))] # only use subset for testing purpose for now

	files = GetFilesFromFolder(output_folder_CG)

	def getGenderForSubstring(substring):
		for r in result:
			if r[0] == substring:
				return r[1] 

	for f in files:
		gender = getGenderForSubstring((f.split('_'))[0])
		if gender == 'male':
			if os.path.isfile(output_folder_CG+"/"+f):
				shutil.copy(output_folder_CG+"/"+f,train_folder+'/male')
				shutil.copy(output_folder_CG+"/"+f,validate_folder+'/male')
		else:
			if os.path.isfile(output_folder_CG+"/"+f):
				shutil.copy(output_folder_CG+"/"+f,train_folder+'/female')
				shutil.copy(output_folder_CG+"/"+f,validate_folder+'/female')
		
	
## FGNET

def DoFGNET():
	csv_file = os.path.join('imageSets/FGNet','annotation.csv')
	output_folder_FG = os.path.join('imageSets/FGNet')
	train_folder = os.path.join('imageSets/FGNet/trunc_image')
	validate_folder = os.path.join('imageSets/FGNet/trunc_image')

	result = []
	with open (csv_file,'r') as csvReader:
		spamreader = csv.reader(csvReader,delimiter=',')
		for row in spamreader:
			result.append(row)

	def GetFilesFromFolder(folder):
		return [f for f in listdir(folder) if isfile(join(folder, f))] # only use subset for testing purpose for now

	def getGenderForSubstringFG(substring):
		print("sub {0}".format(substring))
		for r in result:
			if r[1][0:3] == substring:
				return r[2] 

	files = GetFilesFromFolder(output_folder_FG)

	for f in files: 
		gender = getGenderForSubstringFG((f.split('.'))[0][0:3])
		if gender == 'male':
			if os.path.isfile(output_folder_FG+"/"+f):
				shutil.copy(output_folder_FG+"/"+f,train_folder+'/male')
				shutil.copy(output_folder_FG+"/"+f,validate_folder+'/male')
		else:
			if os.path.isfile(output_folder_FG+"/"+f):
				shutil.copy(output_folder_FG+"/"+f,train_folder+'/female')
				shutil.copy(output_folder_FG+"/"+f,validate_folder+'/female')
	


# FEI 
def doFEI():
	csv_file = os.path.join('imageSets/FEI','FEIAnnotation.csv')
	output_folder_FG = os.path.join('imageSets/FEI/trunc/out')
	train_folder = os.path.join('imageSets/FEI/train')
	validate_folder = os.path.join('imageSets/FEI/validate')

	result = []
	with open (csv_file,'r') as csvReader:
		spamreader = csv.reader(csvReader,delimiter=',')
		for row in spamreader:
			if row != []:
				result.append(row)


	def GetFilesFromFolder(folder):
		return [f for f in listdir(folder) if isfile(join(folder, f))]

	def getGenderForSubstringFEI(substring):
			print("sub {0}".format(substring))
			for r in result:
				if r[0] == substring:
					return r[1] 

	files = GetFilesFromFolder(output_folder_FG)

	for f in files: 
			gender = getGenderForSubstringFEI((f.split('.'))[0])
			print("file: {0} gender: {1}".format(f,gender))
			if gender == 'male':
				if os.path.isfile(output_folder_FG+"/"+f):
					shutil.copy(output_folder_FG+"/"+f,train_folder+'/male')
					shutil.copy(output_folder_FG+"/"+f,validate_folder+'/male')
			else:
				if os.path.isfile(output_folder_FG+"/"+f):
					shutil.copy(output_folder_FG+"/"+f,train_folder+'/female')
					shutil.copy(output_folder_FG+"/"+f,validate_folder+'/female')


# CalTech
def doCalTech():
	csv_file = os.path.join('imageSets/CalTech','CalTEchAnnotation.csv')
	output_folder_FG = os.path.join('imageSets/CalTech/trunc/out')
	train_folder = os.path.join('imageSets/CalTech/train')
	validate_folder = os.path.join('imageSets/CalTech/validate')

	result = []
	with open (csv_file,'r') as csvReader:
		spamreader = csv.reader(csvReader,delimiter=',')
		for row in spamreader:
			if row != []:
				result.append(row)


	def GetFilesFromFolder(folder):
		return [f for f in listdir(folder) if isfile(join(folder, f))]

	def getGenderForSubstringFEI(substring):
			print("sub {0}".format(substring))
			for r in result:
				if r[0] == substring:
					return r[1] 

	files = GetFilesFromFolder(output_folder_FG)

	for f in files: 
			gender = getGenderForSubstringFEI((f.split('.'))[0])
			print("file: {0} gender: {1}".format(f,gender))
			if gender == 'male':
				if os.path.isfile(output_folder_FG+"/"+f):
					shutil.copy(output_folder_FG+"/"+f,train_folder+'/male')
					shutil.copy(output_folder_FG+"/"+f,validate_folder+'/male')
			else:
				if os.path.isfile(output_folder_FG+"/"+f):
					shutil.copy(output_folder_FG+"/"+f,train_folder+'/female')
					shutil.copy(output_folder_FG+"/"+f,validate_folder+'/female')


############## CALL 
#doCK()
#DoFGNET()
#doFEI()
#doCalTech()
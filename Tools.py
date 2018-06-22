## Create Annotation List from images in specific folders
## for this to work images have to be pre-selected in male/female folder, the csv is create from this

# FEI 

import csv
import os
import shutil

from os import listdir
from os.path import isfile, join


dir = os.path.join('imageSets/FEI/train')
csv_out = os.path.join('imageSets/FEI/','FEIAnnotation.csv')

def GetFilesFromFolder(folder):
	return [f for f in listdir(folder) if isfile(join(folder, f))] 

fileList_female = GetFilesFromFolder(dir+"/female")
fileList_male = GetFilesFromFolder(dir+"/male")

print(fileList_female)
print(fileList_male)

list = []
for f_f in fileList_female:
	list.append([ (f_f.split('.'))[0] ,"female"])

for f_m in fileList_male:
	list.append([ (f_m.split('.'))[0],"male"])

with open(csv_out, 'w') as csvFile:
        wr = csv.writer(csvFile)
        wr.writerows(list)

# CalTech


cal_fem = os.path.join('imageSets/CalTech/train/female')
cal_mal = os.path.join('imageSets/CalTech/train/male')

csv_out = os.path.join('imageSets/CalTech/','CalTechAnnotation.csv')


cal_fem_files = GetFilesFromFolder(cal_fem)
cal_mal_files = GetFilesFromFolder(cal_mal)

print(cal_fem_files)
print(cal_mal_files)

list = []
for f_f in cal_fem_files:
	list.append([ (f_f.split('.'))[0] ,"female"])

for f_m in cal_mal_files:
	list.append([ (f_m.split('.'))[0],"male"])

with open(csv_out, 'w', newline='') as csvFile:
    wr = csv.writer(csvFile)
    wr.writerows(list)
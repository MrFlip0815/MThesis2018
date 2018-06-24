
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

# use trunc pictures - they are still in same aspect ratio
image_folder_CK = os.path.join('imageSets/CGPlus/')
image_folder_FG = os.path.join('imageSets/FGNet/')
image_folder_FEI = os.path.join('imageSets/FEI/')
image_folder_CalTech = os.path.join('imageSets/CalTech')

image_folder_full = os.path.join('full')


import pandas as pd

color_male = 'cornflowerblue'
color_female = 'darkorange'
kwon=[]

############################ CHOSE DATA - THIS IS KWON LOBO 6 ###############

#chosenFolder = image_folder_FEI
#chartHeader = "FEI"
#data_female = os.path.join(chosenFolder,"FEI.KWON.LOBO.female.csv")
#data_male = os.path.join(chosenFolder,"FEI.KWON.LOBO.male.csv")

#chosenFolder = image_folder_CK
#chartHeader = "CK"
#data_female = os.path.join(chosenFolder,"CK.KWON.LOBO.female.csv")
#data_male = os.path.join(chosenFolder,"CK.KWON.LOBO.male.csv")

#chosenFolder = image_folder_FG
#chartHeader = "FGNet"
#data_female = os.path.join(chosenFolder,"FG.KWON.LOBO.female.csv")
#data_male = os.path.join(chosenFolder,"FG.KWON.LOBO.male.csv")

#chosenFolder = image_folder_CalTech
#chartHeader = "CalTech"
#data_female = os.path.join(chosenFolder,"CalTech.KWON.LOBO.female.csv")
#data_male = os.path.join(chosenFolder,"CalTech.KWON.LOBO.male.csv")

chosenFolder = image_folder_full
chartHeader = "full"
csv_full = os.path.join(chosenFolder,"KWON.LOBO.FULL.csv")



def DrawFullKwonLobo(list):
	if sys.platform == 'win32':
		data_full = pd.read_csv(csv_full, 
					names = [
					"filename",
					"gender",
					"k1", 
					"k2",
					"k3", 
					"k4", 
					"k5",
					"k6"
					])
		flag_male = data_full["gender"] == 'male'
		data_male = data_full[flag_male]

		flag_female = data_full["gender"] == 'female'
		data_female = data_full[flag_female]

		PlotKwonRatiosForDataSet(data_female,data_male,list,chartHeader)

def DrawKwonLobo6(list):
	if sys.platform == 'win32':
		csv_fem = pd.read_csv(data_female, 
					names = [
					"filename",
					"gender",
					"k1", 
					"k2",
					"k3", 
					"k4", 
					"k5",
					"k6"
					])
		csv_mal = pd.read_csv(data_male, 
					names = [
					"filename",
					"gender",
					"k1", 
					"k2",
					"k3", 
					"k4", 
					"k5",
					"k6"
					])
	PlotKwonRatiosForDataSet(csv_fem,csv_mal,list,chartHeader)



#############################

# remove items from CSV that were chosen in the chosen image folder
# that means, only take CSV row values if there is the real image present (filename in csv)

from scipy.stats import norm
import matplotlib.pyplot as plt

def PlotKwonRatiosForDataSet(data_female, data_male, kwon_list, name_dataset):
	for k in kwon_list:
		val_f = data_female[k]
		val_m = data_male[k]


		val = [1,2,3,2,2,3,4,4,1,9,9,9,9]

		
		mu_f, sigma_f = norm.fit(val_f)
		mu_m, sigma_m = norm.fit(val_m)
		fig, ax = plt.subplots()

		ax.set_title("Ratio-{0} - female vs.male - {1}".format(k[1],name_dataset))
		ax.set_xlabel("Ratio-{0} [1]".format(k[1]))
		ax.set_ylabel("quantity")

		count, bins, ignored = ax.hist(val_f, bins=50, color = color_female, alpha=0.75 )
		count_m, bins_m, ignore_m = ax.hist(val_m, bins=50, color = color_male, alpha=0.75 )
		
		## !!! , nach var -> p_f ,,,,,,
		p_f, = ax.plot(bins, 1/(sigma_f * np.sqrt(2 * np.pi)) *
					   np.exp( - (bins - mu_f)**2 / (2 * sigma_f**2) ),
				 linewidth=2, color=color_female)

		## !!! , nach var -> p_ ,,,,,,
		p_m, = ax.plot(bins_m, 1/(sigma_m * np.sqrt(2 * np.pi)) *
					   np.exp( - (bins_m - mu_m)**2 / (2 * sigma_m**2) ),
				 linewidth=2, color=color_male)

		ax.legend((p_f,p_m),("female","male"), loc="upper right", shadow=True)

	plt.show()

# Standard KWON
#PlotKwonRatiosForDataSet(csv_fem,csv_mal,["k1","k2","k3","k4","k5"],"CK+")
# KWON LOBO 6

#DrawKwonLobo6(["k1","k2","k3","k4","k5","k6"])

DrawFullKwonLobo(["k1","k2","k3","k4","k5","k6"])
#DrawHybrid5()
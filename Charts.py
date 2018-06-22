
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
image_folder_CalTech = os.path.join('imageSets//CalTech')

chosenFolder = image_folder_FG

ck_kwon_data_female = os.path.join(chosenFolder,"FG.KWON.female.csv")
ck_kwon_data_male = os.path.join(chosenFolder,"FG.KWON.male.csv")

color_male = 'cornflowerblue'
color_female = 'darkorange'

import pandas as pd
kwon=[]
if sys.platform == 'win32':
	ck_kwon_fem = pd.read_csv(ck_kwon_data_female, 
				names = [
				"filename",
				"gender",
				"k1", 
				"k2",
				"k3", 
				"k4", 
				"k5"])
	ck_kwon_mal = pd.read_csv(ck_kwon_data_male, 
				names = [
				"filename",
				"gender",
				"k1", 
				"k2",
				"k3", 
				"k4", 
				"k5"])


	
from scipy.stats import norm
import matplotlib.pyplot as plt

def PlotKwonRatiosForDataSet(data_female, data_male, kwon_list, name_dataset):
	for k in kwon_list:
		val_f = data_female[k]
		val_m = data_male[k]

		mu_f, sigma_f = norm.fit(val_f)
		mu_m, sigma_m = norm.fit(val_m)
		fig, ax = plt.subplots()

		ax.set_title("Ratio-{0} Kwon - female vs.male - {1}".format(k[1],name_dataset))
		ax.set_xlabel("Ratio-{0} Kwon [1]".format(k[1]))
		ax.set_ylabel("quantity")

		count, bins, ignored = ax.hist(val_f, 50, normed=True, color = color_female, alpha=0.75 )
		count_m, bins_m, ignore_m = ax.hist(val_m, 50, normed=True, color = color_male, alpha=0.75 )

		# !!! , nach var -> p_f ,,,,,,
		p_f, = ax.plot(bins, 1/(sigma_f * np.sqrt(2 * np.pi)) *
					   np.exp( - (bins - mu_f)**2 / (2 * sigma_f**2) ),
				 linewidth=2, color=color_male)

		# !!! , nach var -> p_ ,,,,,,
		p_m, = ax.plot(bins_m, 1/(sigma_m * np.sqrt(2 * np.pi)) *
					   np.exp( - (bins_m - mu_m)**2 / (2 * sigma_m**2) ),
				 linewidth=2, color=color_female)

		ax.legend((p_f,p_m),("female","male"), loc="upper right", shadow=True)


PlotKwonRatiosForDataSet(ck_kwon_fem,ck_kwon_mal,["k1","k2","k3","k4","k5"],"CK+")

plt.show()

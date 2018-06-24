import pandas as pd 
import os
import sys

annotation_CK = os.path.join('imageSets/CGPlus','annotation.csv')
annotation_FG = os.path.join('imageSets/FGNet','annotation.csv')
annotation_FEI = os.path.join('imageSets/FEI','FEIAnnotation.csv')
annotation_CalTech = os.path.join('imageSets/CalTech','CalTechAnnotation.csv')



result = pd.read_csv(annotation_CalTech, 
			names = [
			"filename",
			"gender"
			])

lbl = result.groupby('gender').size()
print(lbl)



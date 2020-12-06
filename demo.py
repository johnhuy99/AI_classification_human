import cv2 as cv
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# load data from folders
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename),0)
        if img is not None:
            images.append(img)
    return images
folder1="/home/hp/Desktop/classify_object/Human"
folder2="/home/hp/Desktop/classify_object/Non-Human"
images_human = load_images_from_folder(folder1)
images_nonHuman = load_images_from_folder(folder2)

images_human = [image.reshape(images_human[0].shape[0]**2) for image in images_human]
images_human = np.array(images_human)
images_nonHuman = [image.reshape(images_nonHuman[0].shape[0]**2) for image in images_nonHuman]
images_nonHuman = np.array(images_nonHuman)
X = np.concatenate((images_human,images_nonHuman),axis=0)
y_label = np.concatenate((np.ones((120,1)),np.zeros((120,1))),axis=0)



# split data into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

# standalize data set
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

# train the network
def get_Xtrain_ytrain():
	return Xtrain, ytrain

def get_Xtest_ytest():
	return Xtest,ytest






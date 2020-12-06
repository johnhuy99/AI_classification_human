import cv2 as cv
import numpy as np 
import sys
from demo import get_Xtrain_ytrain
from NN_object_detect import NeuralNet

nn = NeuralNet()
Xtrain,ytrain = get_Xtrain_ytrain()
nn.fit(Xtrain,ytrain)

cap = cv.VideoCapture('people.mp4')
fgbg = cv.createBackgroundSubtractorMOG2()

while cap.isOpened:
	_, frame = cap.read()
	fgmask = fgbg.apply(frame)
	blur = cv.GaussianBlur(fgmask.copy(),(5,5),0)
	median = cv.medianBlur(blur,5)

	_, thresh = cv.threshold(median,30, 255, cv.THRESH_BINARY)
	# dilated = cv.dilate(thresh, None, iterations=5)

	contours, hierarchy = cv.findContours(thresh,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	for c in contours:
		if cv.contourArea(c) < 700:
			continue
		(x, y, w, h) = cv.boundingRect(c)
		croped = thresh[y:y+h,x:x+w]
		resized = cv.resize(croped,dsize = (64,64))
		resized = resized.flatten()
		# predict
		k = nn.predict(resized)
		cv.rectangle(frame, (x,y), (x+w-30, y+h-30),(0,255,0),2)
		if k == 1:
			cv.putText(frame, 'human',(x,y),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)


	cv.imshow('rgb',frame)
	

	if cv.waitKey(25) & 0xff == ord('q'):
		break

cap.release()
cv.destroyAllWindows()
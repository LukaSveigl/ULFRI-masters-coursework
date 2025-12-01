import cv2, sys

cascadeFace = cv2.CascadeClassifier("../res/haarcascade_frontalface_default.xml")

filename = sys.argv[1]
img = cv2.imread(filename)

def detectFace(img):
	detectionList = cascadeFace.detectMultiScale(img, 1.05, 5)
	return detectionList
	
def vizualization(img, detectionList):
	for x, y, w, h in detectionList:
		cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
	cv2.imwrite(filename + '.detected.jpg', img)
	
detectionList = detectFace(img)
vizualization(img, detectionList)
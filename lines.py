import numpy as np
import argparse
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

cv2.imshow("images", image)
cv2.waitKey(0)
#image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
image = cv2.fastNlMeansDenoisingColored(image,None,20,10,2,11)
#image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
#image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("images", image)
cv2.waitKey(0)
boundaries = [([0, 115, 100], [255, 255, 255])]
for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
#mask = cv2.fastNlMeansDenoisingColored(mask,None,10,10,7,21)

cv2.imshow("images2", mask)
cv2.waitKey(0)
fp = 0
fb = []
fw = []
threshold = []
widthc = []
h = mask.shape[0]
w = mask.shape[1]
print mask.shape
for i in range(0,w-1):
	fw.append(0)
for i in range(0,h-1):
	for j in range(0,w-1):
		if mask[i,j] == 0:
			fw[j] = 1
			#print "Black Found"
		else:
			fw[j] = 0
		
	if fw.count(1) < 110:
		#print 0
		if fp == 1:
			widthc.append(i)
			fp = 0
#			break
	else:
		#print 1
		fp = 1
print widthc
var = []
for i in range(1,len(widthc)):
	var.append(widthc[i]-widthc[i-1])
avg = reduce(lambda x, y: x + y, var) / len(var)
widthc.insert(0,0)
for p in range(1,len(widthc)):
	if widthc[p] - widthc[p-1] > avg:
		crop_img = mask[widthc[p-1]:widthc[p],0:w-1] # Crop from x, y, w, h -> 100, 200, 300, 400
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		cv2.imshow("cropped", crop_img)
		cv2.waitKey(0)

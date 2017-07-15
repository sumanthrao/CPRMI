import numpy as np
import argparse
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
#boundaries = [([40, 30, 50], [200, 255, 255])]
"""
boundaries = [([0 , 10, 0], [100, 250, 120])]
for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)
boundaries = [([0, 0, 0], [255, 255, 255])]
for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	mask2 = cv2.inRange(image, lower, upper)
	output2 = cv2.bitwise_and(image, image, mask = mask2)
	cv2.imshow("Full", np.hstack([image, output2]))
	cv2.waitKey(0)
boundaries = [([0, 0, 250], [255 , 255, 255])]
for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	mask3 = cv2.inRange(image, lower, upper)
	output3 = cv2.bitwise_and(image, image, mask = mask3)
	cv2.imshow("Full", np.hstack([image, output3]))
	cv2.waitKey(0)
p = cv2.countNonZero(mask)
t = cv2.countNonZero(mask2)
w = cv2.countNonZero(mask3)
#f = (p+0.0)/t * 100
#r = (w+0.0)/t * 100
f = (p+0.0)/(t-w) * 100
#print p
#print t
#print f
print f
"""
boundaries = [([0, 120, 0], [255, 255, 255])]
for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)
cv2.imshow("images", image)
cv2.imshow("images2", mask)
cv2.waitKey(0)
fp = 0
fb = []
fw = []
widthc = []
h = mask.shape[0]
w = mask.shape[1]
for i in range(0,h-1):
	fw.append(0)
for i in range(0,w-1):
	for j in range(0,h-1):
		if mask[j,i] == 0:
			fw[j] = 1
			#print "Black Found"
		else:
			fw[j] = 0
	if fw.count(1)<6:
		#print 0
		if fp == 1:
			widthc.append(i)
			fp = 0
#			break
	else:
		#print 1
		fp = 1
print widthc
widthc.insert(0,0)
for p in range(1,len(widthc)):
	crop_img = mask[0:h-1,widthc[p-1]:widthc[p]] # Crop from x, y, w, h -> 100, 200, 300, 400
	# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
	cv2.imshow("cropped", crop_img)
	cv2.waitKey(0)

import numpy as np
import argparse
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
#boundaries = [([40, 30, 50], [200, 255, 255])]
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

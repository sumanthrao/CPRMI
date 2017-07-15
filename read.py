import numpy as np
import argparse
import cv2
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import os
os.system("rm lines/*")
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])

cv2.imshow("images", image)
cv2.waitKey(0)
original = image
#image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
image = cv2.fastNlMeansDenoisingColored(image,None,20,10,2,11)
#image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
#image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("images", image)
cv2.waitKey(0)
boundaries = [([0, 90, 0], [255, 255, 255])]
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
minm = []
color = []
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
	color.append(sum(fw))
y = np.array(color)
y = savitzky_golay(y, 51, 3)
x = []
for i in range(0,len(y)):
	x.append(i)

# this way the x-axis corresponds to the index of x
plt.plot(x, y)
plt.show()
y = np.array(y)
maxm = argrelextrema(y, np.greater)  # (array([1, 3, 6]),)
minm = argrelextrema(y, np.less)  # (array([2, 5, 7]),)
print minm
print maxm
minm = minm[0].tolist()

var = []
for i in range(1,len(minm)):
	var.append(minm[i]-minm[i-1])
avg = reduce(lambda x, y: x + y, var) / len(var)

minm.insert(0,0)
i = 0
for p in range(1,len(minm)):
	if minm[p] - minm[p-1] > 0.6*avg:
		crop_img = original[minm[p-1]:minm[p],0:w-1] # Crop from x, y, w, h -> 100, 200, 300, 400
		# NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
		i+=1
		cv2.imshow("cropped", crop_img)
		cv2.waitKey(0)
		cv2.imwrite("lines/"+str(i)+".png",crop_img)

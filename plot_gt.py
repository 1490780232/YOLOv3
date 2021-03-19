import cv2
im = cv2.imread("1.bmp")
h, w , _ = im.shape
x = int(w * 0.5242187500000001)
y = int(h * 0.5104166666666666)
w_i = int((w * 0.37031250000000004)/2)
h_i = int((h * 0.42916666666666664)/2)
im = cv2.rectangle(im, (x-w_i, y-h_i), (x+w_i, y+h_i), (0, 255, 0), 2)
cv2.imshow("", im)
cv2.waitKey()
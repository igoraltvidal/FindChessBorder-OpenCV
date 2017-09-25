import numpy as np
import cv2

cap = cv2.VideoCapture('cow_video.webm')

def PointSurroundMeanColor(mean_point, image, mean_variation, RGB_index):
     color_sum = image[mean_point[1] + mean_variation, mean_point[0] + mean_variation][RGB_index] 
     print("POINT FUNCTION 1")
     print(color_sum)
     #color_sum = color_sum +  image[mean_point[1] + mean_variation, mean_point[0] - mean_variation][RGB_index] 
     #color_sum = color_sum +  image[mean_point[1] - mean_variation, mean_point[0] - mean_variation][RGB_index]
     #color_sum = color_sum +  image[mean_point[1] - mean_variation, mean_point[0] + mean_variation][RGB_index]
     color_mean = color_sum/1
     return color_mean
     
def CountourMeanColor(square_array, img, RGB_index, variation):
     if abs(square_array[0][0][0] -  square_array[1][0][0]) > 20 and  abs(square_array[1][0][1] -  square_array[2][0][1]) > 20 and \
        abs(square_array[2][0][0] -  square_array[3][0][0]) > 20 and  abs(square_array[3][0][1] -  square_array[0][0][1]) > 20:
	     color_sum = img[square_array[0][0][1] + variation, square_array[0][0][0] + variation][RGB_index] + \
		     img[square_array[1][0][1] + variation, square_array[1][0][0] - variation][RGB_index] + \
		     img[square_array[2][0][1] - variation, square_array[2][0][0] - variation][RGB_index] + \
		     img[square_array[3][0][1] - variation, square_array[3][0][0] + variation][RGB_index] + \
		     img[(square_array[0][0][1] +  square_array[2][0][1])/2,  (square_array[0][0][0] +  square_array[2][0][0])/2][RGB_index]
	     color_mean = color_sum/5
	     return color_mean

while(cap.isOpened()):

    ret, img = cap.read()
    imagen = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh1 = cv2.adaptiveThreshold(imagen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
		    cv2.THRESH_BINARY,11,2)

    im2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)


    tagCntVec = []

    for cnt in contours:

        square_left_located = False
        square_right_located = False

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if cv2.contourArea(approx) > 10 and len(approx) == 4:

            if 0 < CountourMeanColor(approx,img,0,10) < 90 and 0 < CountourMeanColor(approx,img,1,10) < 90 and 0 < CountourMeanColor(approx,img,2,10) < 90:
                 mean_point = [abs(approx[0][0][0] +  approx[2][0][0])/2, (approx[0][0][1] +  approx[2][0][1])/2]
		 mean_point_right = [mean_point[0] + abs(approx[0][0][0] -  approx[1][0][0]), mean_point[1]]
		 mean_point_left = [abs(approx[0][0][0] +  approx[2][0][0])/2 - abs(approx[0][0][0] -  approx[1][0][0]), (approx[0][0][1] +  approx[2][0][1])/2]

		 if 160 < PointSurroundMeanColor(mean_point_right, img, 0,0) < 255 and 160 < PointSurroundMeanColor(mean_point_right, img, 0,1) < 255 and 160 <     	          PointSurroundMeanColor(mean_point_right, img, 0,2) < 255:
	              cv2.circle(img,(mean_point_right[0],mean_point_right[1]), 2, (0,0,255), -1)
		      square_right_located = True

		 if 160 < PointSurroundMeanColor(mean_point_left, img, 0,0) < 255 and 160 < PointSurroundMeanColor(mean_point_left, img, 0,1) < 255 and 160 <     	          PointSurroundMeanColor(mean_point_left, img, 0,2) < 255:
		      cv2.circle(img,(mean_point_left[0],mean_point_left[1]), 2, (0,0,255), -1)
		      square_left_located = True

		 if square_right_located == True and square_left_located == True:
		      radious = abs(mean_point_left[0] - mean_point_right[0])
		      cv2.circle(img,(mean_point[0],mean_point[1]), radious , (0,0,255), 2)

		      tagCntVec.append(approx)
		      cv2.drawContours(img, tagCntVec, -1, (128,255,0), 1)

		      cv2.imshow('frame',img)
		      if cv2.waitKey(1) & 0xFF == ord('q'):
		          break

cap.release()
cv2.destroyAllWindows()


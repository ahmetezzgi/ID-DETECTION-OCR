from PIL import Image
import pytesseract
import cv2
import os
import numpy as np
import imutils
import time
import datetime

cam = cv2.VideoCapture(0)

while True:

    ret,image = cam.read()
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # this calculate the histogram of the image you input
    # if this is under/below a certain value (which depend of the colors in the image), a certain thresh will be choosed among another
    hist, bins = np.histogram(hsv.ravel(), 256, [0, 256])
    #print(hist[-1])

        # binary
    ret, thresh = cv2.threshold(hsv[:, :, 0], 55, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
       

        # dilation
    kernel = np.ones((1, 1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        #cv2.imshow('dilated', img_dilation)
        

        # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    #if hist[-1] < 25000:

    for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
        roi = image[y:y + h, x:x + w]

            # show ROI
            # cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.waitKey(0)

        if w > 400 and h > 300:
            out = 'outs\\roi{}.png'.format(i)
            cv2.imwrite(out,roi)
            text = pytesseract.image_to_string(Image.open(out),lang='eng')

            os.remove(out)
                
            things = text.split('<')

            now = datetime.datetime.now()

            nolar = []

            for i in range(len(things)):
                if (len(things[i]) == 11):
                    nolar.append(things[i])
                        

            kimlik_no = 0
            sum_tekhane = 0
            sum_cifthane = 0
            sum_onhane = 0

            for i in nolar:
                print('secili no : '+i)
                for x in range(len(i)-2):
                    try:
                    	if x % 2 == 0:
                        	sum_tekhane += int(i[x])
                    	else:
                        	sum_cifthane += int(i[x])
                    except ValueError:
                        print('dizide sayi harici karakter var')
                        print('\n-----------------------')
                        break
           
                kosul1 = (sum_tekhane * 7) - sum_cifthane

                #print(kosul1)

                for x in range(len(i)-1):
                	try:
                		sum_onhane += int(i[x])
                	except:
                		print('dizide sayi harici karakter var')
                		print('\n------------------------')
                		break

                if str(kosul1 % 10) == i[-2] and str(sum_onhane % 10) == i[-1]:
                    kimlik_no = i
                    print(kosul1)
                    print(sum_onhane)
                    print('TC bulundu\n' + kimlik_no)
                    f = open('tc_nolar.txt','a')
                    f.write(str(kimlik_no + ' --- ' + now.strftime("%Y-%m-%d %H:%M") + '\n'))
                    print('\n-----------------------')
                    bulunan = 'fotolar\\{}.png'.format(i)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image,'TC BULUNDU',(100,200), font, 2,(0,0,255),2,cv2.LINE_AA)
               			#font = cv2.FONT_HERSHEY_SIMPLEX
 										#cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
                    cv2.imwrite(bulunan,roi)
                else:
                    print(kosul1)
                    print(sum_onhane)
                    sum_onhane = 0
                    sum_tekhane = 0
                    sum_cifthane = 0
                    print('TC bulunamadi')
                    #print(sum_onhane)
                    print('\n-----------------------')
                        

    cv2.imshow('marked areas', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
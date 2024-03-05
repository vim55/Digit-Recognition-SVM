import cv2
import os
import train
import numpy as np
from os import listdir
import sys

#un-comment this if you need to re-train the svm classifier
#train.train_svm()

# initialise file diretory path for testing image
TASK_FILES = ""

# output directory 
TASK_OUTPUT = os.getcwd() + "/output/"

# Take an image of a building sign and get the numbers
def task(img, imgNum):
    img_copy = np.copy(img)
    img_copy_c = np.copy(img)
    #remove red channel
    img_copy[:,:,2] = np.zeros([img_copy.shape[0], img.shape[1]])
    #convert to gray scale image 
    grey = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    
    #blur image
    blur = cv2.bilateralFilter(grey,13,75,75)
    
    # Thresholding with the adaptive thresholding method
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=21, C=1)
    #apply more filter to make the building plate frame more clear
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.medianBlur(thresh,5)

    # Compute all the contours on the inverted thresholded image
    _, contours, heir = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Allocate a str for building number to avoid duplicates
    tempStr = ""
    # Loop over all the contours,only those countour meet the Area and Aspect Ratio are
    # feeded to get_plat_digit for digit recognition
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Get the x, y coordinates and width and height of a bounding box for this contour
        x, y, w, h = cv2.boundingRect(cnt)  
        # apply constraint to filter the contours
        if 1600 <= area <= 17645 or 17647 <= area <= 22446 or 22448 <= area <= 30000 or 73000 <= area:    
            if  0.4 <= h/w <=  1.1:
                building_plate = img_copy_c[y : y + h, x : x + w]
                if str(imgNum) == "08":
                    display_img("val",building_plate)
                # get the recognized digit and the actual building plate location(50 percent ground truth)
                plate_num_str, plat_size, draw = get_plate_digit(building_plate,imgNum)
                  
                # if the buidling number reconized in string is not empty
                # than output the building plate and the number recognized
                if plate_num_str != "empty":
                    ratio = plat_size[3]/plat_size[2]
                    # only write the building number string that has longest number and the height vs width ratio is less than 1.9
                    if tempStr != plate_num_str and len(plate_num_str) > len(tempStr) and ratio < 1.9:
                        x, y, w, h = plat_size[0],plat_size[1],plat_size[2],plat_size[3]
                        # get the actual plate that complies to 50 percent ground truth
                        actual_plate = draw[y : y + h, x : x + w]
                        
                        # write out the buidling plate images in jpg
                        cv2.imwrite(TASK_OUTPUT + "DetectedArea" + str(imgNum) + ".jpg", actual_plate)

                        # write out the bounding box x,y,w,h value in txt
                        f = open(TASK_OUTPUT + "BoundingBox" + str(imgNum) + ".txt", "w")
                        f.write("x:" + str(x) + " " + "y:" + str(y) + " " + "w:" + str(w) + " " +"h:" + str(h) + " ")
                        f.close()

                        # write out the house number recognized in txt
                        f = open(TASK_OUTPUT + "House" + str(imgNum) + ".txt", "w")
                        f.write("Building " + plate_num_str)
                        f.close()
                        tempStr = plate_num_str



# this function reads the plate and image number
# and output recognize the digit within plate and the plate location
def get_plate_digit(plate, imgNum):
    # Load the pre trained model
    svm = cv2.ml.SVM_load("svm_classifier.model")  

    #make a copy of the plate image
    plate_copy = np.copy(plate)
    plate_copy_draw = np.copy(plate)
    
    # gray scale and threshold the plate image
    plate_gray = cv2.cvtColor(plate_copy, cv2.COLOR_BGR2GRAY)  
    thresh = cv2.inRange(plate_gray, 150, 255)  

    #find the digit countour within the bulding plate
    _, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    digit_bb_list = list()
    i = 0
    averageArea = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        # constraint the area to get the digit contours
        if 73 <= area <= 74 or 100 <= area <= 8000 or 18953 <= area <= 18954 or 19848 <= area <= 19850 or 38800 <= area <= 38900:
            if  1.31 <= h/w <= 3.4:
                digit_bb_list.append(cv2.boundingRect(cnt))
                
                #variable used to calculate the average area for all the countours  
                averageArea = averageArea + area
                i = i + 1

    # initlize temporary x, y, width and height for locating actual buidling plate loaction 
    # this is done by using the location of the digits recognized 
    tempx, tempTopY,tempBotY, tempw, temph = 0 , 0 , 0 , 0 , 0

    # if area is not zero than the digit in the plate is been reconigzed 
    if averageArea > 0.0:
        averageArea = averageArea / i
        # clean unwanted contour using average area
        digit_bb_list = clean_digit_contours(digit_bb_list, averageArea, i)
        
        # sort the digit using x cordiniate value
        digit_bb_list.sort(key=lambda c: c[0])
        # set the initial bounding box for the actual building plate 
        tempx, tempTopY, tempw, temph = digit_bb_list[0][0], digit_bb_list[0][1], digit_bb_list[0][2],digit_bb_list[0][3]
        tempBotY = tempTopY + temph

    plate_num_str = ""
    plate_actual_size = list()

    for i in range(len(digit_bb_list)):
        # assign the x, y, width and height value of the digit bounding box from the sorted digit list
        x, y, w, h = digit_bb_list[i][0], digit_bb_list[i][1], digit_bb_list[i][2],digit_bb_list[i][3]
        # crop the digit from the plate image
        digit_number = plate_copy[y : y + h, x : x + w]
        # draw this digit bounding box in the plate image
        digit_draw = cv2.rectangle(plate_copy_draw, (x,y), (x+w,y+h), (0,255,0), 2)

        # get the minimum x cordinate for the atual building plate size
        if tempx > x:
            tempx = x
        # get the most top y coordinate by comparing all digit contours
        if tempTopY > y:
            tempTopY = y 
        # get the most bottom y coording by comparing all digit contours
        if tempBotY < y+h:
            tempBotY = y+h
        # check if current digit is the last digit in the list 
        if i == (len(digit_bb_list) - 1):
            # get the width of the building plates using last digit x cordinate
            lastDigit_x_val = x + w
            tempw = lastDigit_x_val - tempx
            # get the height of the building plates by calculates botY - topY
            temph = tempBotY - tempTopY  
        
        
        # add 2 black pixels boarder to each digit on all sides
        img_add_boarder = cv2.copyMakeBorder(digit_number, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
        # gray scale the digit image
        img_add_boarder = cv2.cvtColor(img_add_boarder, cv2.COLOR_BGR2GRAY)
        # threshhold the digit image (same thresh value used when training svm)
        _, img_add_boarder = cv2.threshold(img_add_boarder, 150, 255, cv2.THRESH_BINARY)

        # Resize the digit to required dimensions for hog calculation
        dgt_resized = cv2.resize(img_add_boarder, (28, 40))  
        # Find the hog decriptor and predict using trained svm model 
        hog_im = train.hog(dgt_resized)  
        dgt_resized = np.array(hog_im, dtype='float32').reshape(-1, 64)
        result = svm.predict(dgt_resized)[1].ravel()
        #append predicted digit into string
        plate_num_str += str(int(result[0]))

    # get the final building plate location cordinates into list
    plate_actual_size.append(tempx)
    plate_actual_size.append(tempTopY)
    plate_actual_size.append(tempw)
    plate_actual_size.append(temph)

    if len(plate_num_str) >= 1:   
        #return the build number string and plate location cordinates 
        return plate_num_str , plate_actual_size , digit_draw
    else:
        # return empty string when nothing got reoconized in the given image
        return "empty" , plate_actual_size , None


# this function clean the digit countour using average area, 
# since digit should be largest area detected within the plate 
def clean_digit_contours(digit_list, areaThresh, numCnt):
    new_digit_list = list()
    # only clean when there is atleast one contour
    if numCnt >= 1:
        for i in range(0, len(digit_list)):
            _, _, w, h = digit_list[i][0], digit_list[i][1], digit_list[i][2], digit_list[i][3]
            area = w*h
            if area >= areaThresh * 0.9 :
                new_digit_list.append(digit_list[i])

    return new_digit_list


# this function serves convenient purpose to display images 
def display_img(window_name, image, close=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    if close:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # check for valid arguments 
    if(len(sys.argv) != 2):
        print("error: please put correct file path in the arguments")
        print("e.g.\n python3 task.py /home/User/Documents/assignment/train_updated/")
    else: 
        #assign file path from the command line argument
        TASK_FILES = sys.argv[1]

        # For each run empty the output folder first in output directory
        for file in os.listdir(TASK_OUTPUT):
            os.remove(TASK_OUTPUT + file)

        # perform the recognising task according to image file name
        all_files = listdir(TASK_FILES)
        # loop through testing files 
        for file in all_files:
            if file.startswith("tr") and file.endswith(".jpg"):
                number = file[2:-4]
                img = cv2.imread(TASK_FILES + file)
                if img is not None:
                    print("processing imageNo " + number)
                    task(img, number)

            elif file.startswith("val") and file.endswith(".jpg"):
                number = file[3:-4]
                img = cv2.imread(TASK_FILES + file)
                if img is not None:
                    print("processing imageNo " + number)
                    task(img, number)
            
            elif file.startswith("test") and file.endswith(".jpg"):
                number = file[4:-4]
                img = cv2.imread(TASK_FILES + file)
                if img is not None:
                    print("processing imageNo " + number)
                    task(img, number)

        print("Complete - check output folder for results")
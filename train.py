import numpy as np
import cv2
import os
from os import listdir

#training folder direcotry to train svm
# e.g for Digits folder that store in root directory you do following
#TRAINING_DIGITS_FILES = os.getcwd() + "/Digits/"
TRAINING_DIGITS_FILES = "/home/student/train/"

# This function maps the file name to numeric value which is used as the target value during the training of SVM
def get_label(fname):
    labels = ['digit0', 'digit1', 'digit2', 'digit3', 'digit4', 'digit5', 'digit6', 'digit7', 'digit8', 'digit9']
    for i in range(len(labels)):
        if labels[i] in fname:
            return i

# This function finds the HOG
# code reference: https://docs.opencv.org/master/dd/d3b/tutorial_py_svm_opencv.html
def hog(img):
    number_of_bins = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)  # gradient x
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)  # gradient y
    mag, ang = cv2.cartToPolar(gx, gy)  # Combine two gradients get magnitude and angle
    bins = np.int32(number_of_bins * ang / (2 * np.pi))  # quantizing binvalues in (0...16)

    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]  # Image is divided into 4 squares
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]  # Image is divided into 4 squares

    list_of_histogram = list()

    # This loop calculates the histogram of each big square of the image
    for bin_cell, mag_cell in zip(bin_cells, mag_cells):
        bin_count = np.bincount(bin_cell.ravel(), mag_cell.ravel(), number_of_bins)
        list_of_histogram.append(bin_count)

    # Change the histogram into 16 X 4 = 64 dimension vector
    hist = np.hstack(list_of_histogram)  
    return hist

# This function train the svm and save the model
def train_svm():
    data, targets = load_digits_data(TRAINING_DIGITS_FILES)
    svm = cv2.ml.SVM_create()  # Create the model
    svm.setKernel(cv2.ml.SVM_INTER)  # Choose the filter
    svm.train(data, cv2.ml.ROW_SAMPLE, targets)  # Train the SVM using each Row of the data set as one sample
    svm.save('svm_classifier.model')  # Save the trained model in current directory
    
def display_img(window_name, image, close=True):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    if close:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# This function reads the digits images into data set required for SVM training
def load_digits_data(files_path):
    digits_data = list()
    digits_target = list()
    # find all digit files 
    list_digit_files = listdir(files_path)
    # loop through digit files and convert them into data for the SVM training
    for digit_files in list_digit_files:
        digit_imgs = listdir(files_path + digit_files + "/")
        for digit_img_name in digit_imgs:
            digit_img_path = files_path + digit_files + "/" + digit_img_name
            img = cv2.imread(digit_img_path, cv2.IMREAD_GRAYSCALE)  
            #resize to img to 24 by 40 so hog descriptor is consistence 
            img_resize = cv2.resize(img, (28, 40))  
            # thresh holding
            _, thresh = cv2.threshold(img_resize, 150, 255, cv2.THRESH_BINARY)  
            # Find the HOG 
            hog_img = hog(thresh)  
            #get the numeric value lable according to file name
            img_label = get_label(digit_img_path)  

            digits_data.append(hog_img)
            digits_target.append(img_label)   
    # Final data set
    return np.array(digits_data, dtype='float32'), np.array(digits_target)  

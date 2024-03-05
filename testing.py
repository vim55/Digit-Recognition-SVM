import os
from os import listdir

# output directory 
TASK_OUTPUT = os.getcwd() + "/output/"

train = ["729","304","210","211","42","28","108","100","302","32","420","2202","23","2","123","1203","293","37","85","124","32","60","5","237","23"]
val = ["48", "35", "94", "302", "71", "26"]


def testAccuracy(test):
    all_files = listdir(TASK_OUTPUT)
    
    number_files = len(test)
    correct = 0
    for file in all_files:
        if file.startswith("House") and file.endswith(".txt"):
            number = file[5:-4]

            f = open(TASK_OUTPUT + "House" + str(number) + ".txt", "r")
            line = f.readline()
            f.close()
            buidling_number = line[9:]
            i = int(number) - 1

            if str(test[i]) == str(buidling_number):
                correct = correct + 1
            else:
                print("error building" + number + " expected " + test[i] + " output " + str(buidling_number) )
        
    accuracy = correct/number_files * 100
    print("----------------------")
    print("total number: " + str(number_files))
    print("correct prediction: " + str(correct))
    print("accuracy: " + str(accuracy) + "%")


if __name__ == "__main__":
    testAccuracy(train)
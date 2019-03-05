import cv2
import numpy as np 
from pathlib import Path #for looping throu frames in folder
import glob
### video implementation

R_t = 130 # Red component threshhold
S_t = 60 #saturation threshhold
L1_t = 200 #light grey lower threshhold
L2_t = 255 #light grey upper threshhold
D1_t = 178 #dark grey lower bound
D2_t = 200 #dark grey upper bound

def R_component(r):
    if(r > R_t):
        return 1
    else:
        return 0

def RGB_compare(b, g, r):
    if (r >= g > b):
        return 1
    else:
        return 0
    
def saturation(s, r):        
    lim = ((255.0-r)*S_t/R_t)
    if(s >= lim):
        return 1
    else:
        return 0

def is_fire_pixel(b, r, g, s):
    if(R_component(r) and RGB_compare(b, g, r) and saturation(s, r)):
        return 1
    else:
        return 0

def is_grey(b, r, g):
    if(abs(b-r)<=30 and abs(r-g)<=30 and abs(g-b)<=30):
        return 1
    else:
        return 0

def grey_intensity(v):
    if(L1_t <= v <= L2_t or D1_t <= v <= D2_t):
        return 1
    else:
        return 0

def is_smoke_pixel(b, r, g, v):
    if(is_grey(b, r, g) and grey_intensity(v)):
        return 1
    else:
        return 0
        
true_positive = 0       #real fire is real
false_positive = 0      #false fire is real
true_negative = 0     #fasle fire is false
false_negative  = 0   #real fire is fasle
#cap = cv2.VideoCapture('./nist_fire_ds/Room_corner_fires__Influence_of_distance_from_corner_on_flame_height_Video_Download.mp4')
#_, frame = cap.read()
for iter in range(2): #to iterate on two folders 
    print("i="+str(iter))
    if(iter == 0):
        path = "/home/tharagesh/code_stuff/fireDetection/frames/homefire2/nofire"
    else:
        path = "/home/tharagesh/code_stuff/fireDetection/frames/homefire2/fire"

    frame = cv2.imread(path+"/1.jpg")
    
    one = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
    one_cnt = 0
    two = np.ones((frame.shape[0], frame.shape[1], frame.shape[2]))
    two_cnt = 0
    filenames = glob.glob(path+"/*.jpg")
    filenames.sort()
    for imagepath in filenames:
        #take each frame in RBG format
        print(imagepath)
        frame = cv2.imread(imagepath)
        #_, frame = cap.read()
        #print(_)
        #convert BGR to HSV
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        img_fp = np.zeros((frame.shape[0], frame.shape[1])) #final fire pixel array
        img_sp = np.zeros((frame.shape[0], frame.shape[1])) #final smoke pixel array
        fire_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
        smoke_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))

        tre_cnt = 0

        for i in range(frame.shape[0]): #row iterate
            for j in range(frame.shape[1]): #col iterate
                bgr = frame[i][j].astype(np.float)
                hsv = img_hsv[i][j].astype(np.float)
                blue = bgr[0]
                green = bgr[1]
                red = bgr[2]
                satur = hsv[1]
                intens = hsv[2]
                #fire pixel detection
                if(is_fire_pixel(blue, red, green, satur)):
                    img_fp[i][j] = 1
                    fire_pixel[i][j] = bgr
                    tre_cnt += 1

                #smoke pixel detection
                if(is_smoke_pixel(blue, red, green, intens)):
                    img_sp[i][j] = 1
                    smoke_pixel[i][j] = bgr

        tre = fire_pixel 
        tre_cnt = tre_cnt

        FD_t1 = np.absolute(np.subtract(tre, two))
        #print(tre)
        FD_t = np.absolute(np.subtract(two, one))
        FD = np.divide(np.absolute(FD_t1-FD_t), FD_t, out=np.zeros_like(np.abs(FD_t1-FD_t)), where=FD_t!=0)
        print(np.amax(FD))
        per = float((FD>64.0).sum())/FD.size
        #num = max(one_cnt, two_cnt, tre_cnt)
        #mean_val = np.divide(FD.sum(), num)
        print("FD>64.0sum : ", (FD>64.0).sum())
        print("FD size   : ", FD.size)
        print("percentage: ", per)

        if(per >= 0.00001):
            print("Real flame")
            if(iter == 0):
                false_positive += 1
                #print("hrhrhrhrhrhrhrh")
            else:
                true_positive += 1
                #print("dalsefalse")
        else:   
            print("Fake flame")
            if(iter == 0):
                true_negative += 1
                #print("hrhrhrhrhrhrhrh")
            else:
                false_negative += 1
                #print("dalsefalse")
        one = two
        two = tre

        #cv2.imshow('frame', frame)
        #cv2.imshow('img_fp', img_fp)
        #cv2.imshow('img_sp', img_sp)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

print("========================performance metrics============================")
print("#true_positive    #real fire is detected as real   : " + str(true_positive))
print("#false_positive   #false fire is detected as real  : " + str(false_positive))
print("#true_negative    #fasle fire is detected as false : " + str(true_negative))
print("#false_negative   #real fire is detected as fasle  : " + str(false_negative))

accuracy = float(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative)
print("Accuracy   :" + str(accuracy))
precision = float(true_positive)/(true_positive+false_positive)
print("Precision  :" + str(precision))
recall = float(true_positive)/(true_positive+false_negative)
print("Recall     :" + str(recall))
errorrate = float(false_negative+false_positive)/(true_positive+true_negative+false_positive+false_negative)
print("Error rate :" + str(errorrate))
f1score = 2.0*(recall*precision)/(recall+precision)
print("F1 score   :" + str(f1score))
far = float(false_positive)/(false_positive+true_negative)
print("FAR        :" + str(far))
frr = float(false_negative)/(false_negative+true_positive)
print("FRR        :" + str(frr))
'''
print("Accuracy   :" + str(accuracy))
print("Error rate :" + str(errorrate))
print("Precision  :" + str(precision))
print("Recall     :" + str(recall))
print("F1 score   :" + str(f1score))
print("FAR        :" + str(far))
print("FRR        :" + str(frr))
'''
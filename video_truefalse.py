import cv2
import numpy as np 
import matplotlib.pyplot as plt

### video implementation

R_t = 130 # Red component threshhold
S_t = 60 #saturation threshhold
L1_t = 200 #light grey lower threshhold
L2_t = 255 #light grey upper threshhold
D1_t = 178 #dark grey lower bound
D2_t = 200 #dark grey upper bound

def R_component(r): # Comparing the red component with the given threshold
    if(r > R_t):
        return 1
    else:
        return 0

def RGB_compare(b, g, r): # Comparing the red componenet values with green and blue
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

def is_fire_pixel(b, r, g, s):  # Chromatic analysis
    if(R_component(r) and RGB_compare(b, g, r) and saturation(s, r)):
        return 1
    else:
        return 0

def is_grey(b, r, g):
    if(abs(b-r)<=30 and abs(r-g)<=30 and abs(g-b)<=30):
        return 1
    else:
        return 0

def grey_intensity(v):      # Checking for the intensity level for smoke detection
    if(L1_t <= v <= L2_t or D1_t <= v <= D2_t):
        return 1
    else:
        return 0

def is_smoke_pixel(b, r, g, v):     # Checking for smoke pixel
    if(is_grey(b, r, g) and grey_intensity(v)):
        return 1
    else:
        return 0
        

cap = cv2.VideoCapture('E:/General/Personal/Pro/DST 2017/Fire detection/dst/sdf/videos/videojapan.mp4')  # Reading the video file
_, frame = cap.read()
one = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))        # First frame 
one_cnt = 0
two = np.ones((frame.shape[0], frame.shape[1], frame.shape[2]))         # Second frame
two_cnt = 0
x=-1 #x axis
while(1):
    #take each frame in RBG format
    _, frame = cap.read()
    if _==False:    # If Frame is not read properly , them breaks
        break
    x+=1
    #print(_)
    #convert BGR to HSV
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    img_fp = np.zeros((frame.shape[0], frame.shape[1]),dtype=np.uint8) #final fire pixel array
    img_sp = np.zeros((frame.shape[0], frame.shape[1]),dtype=np.uint8) #final smoke pixel array
    fire_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))     # To keep track of Fire pixels
    smoke_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))    # To keep track of Smoke pixels

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
                img_fp[i][j] = 1    # Changing in Grayscale image
                fire_pixel[i][j] = bgr  # Storing the fire pixels
                tre_cnt += 1        # Count of Fire pixels
                

            #smoke pixel detection
            if(is_smoke_pixel(blue, red, green, intens)):
                img_sp[i][j] = 1
                smoke_pixel[i][j] = bgr

    tre = fire_pixel
    tre_cnt = tre_cnt
    plt.scatter(x,tre_cnt,alpha=0.5,color='blue')   # Scattering the fire pixel popints
    FD_t1 = np.absolute(np.subtract(tre, two))  # Calculate the Fire disorder between 2-3 
    #print(tre)
    FD_t = np.absolute(np.subtract(two, one))   # Calculate the Fire disorder between 1-2
    FD = np.divide(np.absolute(FD_t1-FD_t), FD_t, out=np.zeros_like(np.abs(FD_t1-FD_t)), where=FD_t!=0)     # Checking with Fire disorder threshold value
    print(np.amax(FD))      # Printing maximum fire disorder value
    per = float((FD>64.0).sum())/FD.size    
    #num = max(one_cnt, two_cnt, tre_cnt)
    #mean_val = np.divide(FD.sum(), num)
    print("FD>64.0sum : ", (FD>64.0).sum())
    print("FD size   : ", FD.size)
    print("percentage: ", per)
    if(per >= 0.00001):         # Check with threshold value
        print("Real flame")
    else:   
        print("Fake flame")

    one = two
    two = tre
    '''
    cv2.imshow('frame', frame)      # Display video file
    cv2.imshow('img_fp', img_fp)    # Display Fire pixels
    cv2.imshow('img_sp', img_sp)    # Display smoke pixels
    '''
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

plt.xlabel('Frame')     # Set xlabel for graph
plt.ylabel('No of fire pixels') # Set ylabel for graph
plt.show()  # Print graph
cap.release()
cv2.destroyAllWindows()

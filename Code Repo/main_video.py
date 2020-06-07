import cv2
import numpy as np 

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
        

cap = cv2.VideoCapture('E:/General/Personal/Pro/DST 2017/Fire detection/dst/sdf/videos/videojapan.mp4')
_, frame = cap.read()

one = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
two = np.ones((frame.shape[0], frame.shape[1], frame.shape[2]))
me=0
re=0
fa=0

while(1):
    #take each frame in RBG format
    _, frame = cap.read()
    #print(_)
    if(_==True):
        #convert BGR to HSV
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        img_fp = np.zeros((frame.shape[0], frame.shape[1])) #final fire pixel array
        img_sp = np.zeros((frame.shape[0], frame.shape[1])) #final smoke pixel array
        fire_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))
        smoke_pixel = np.zeros((frame.shape[0], frame.shape[1], frame.shape[2]))

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

                #smoke pixel detection
                if(is_smoke_pixel(blue, red, green, intens)):
                    img_sp[i][j] = 1
                    smoke_pixel[i][j] = bgr

            tre = fire_pixel

            FD_t1 = np.absolute(np.subtract(tre, two))
            #print(tre)
            FD_t = np.absolute(np.subtract(two, one))
            FD = np.divide(np.absolute(FD_t1-FD_t), FD_t, out=np.zeros_like(np.abs(FD_t1-FD_t)), where=FD_t!=0)
            #print(np.mean(FD))
            me=me+np.mean(FD)
            if(np.mean(FD) >= 0.9):
                print("Real flame")
                re=re+1
            else:   
                print("Fake flame")
                fa=fa+1

            one = two
            two = tre

    
'''
            cv2.imshow('frame', frame)
            cv2.imshow('img_fp', img_fp)
            cv2.imshow('img_sp', img_sp)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
'''
cv2.destroyAllWindows()
print("REAL ",re)
print("FAKE ",fa)
print(np.mean(me))



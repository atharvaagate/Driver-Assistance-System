
import cv2
import numpy as np
import matplotlib.pyplot as plt



def get_first_frame(video) :
    road_video = cv2.VideoCapture(video)
    
    while(road_video.isOpened()):
        print("inside")
    
        _,frame = road_video.read() #replace lane_image with frame
        if _ == True:
           
            #frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_AREA)
            #print(frame)
            return frame
    

def find_roi(video, plot = False) :
    #to find region of interest point
    frame = get_first_frame(video)
    height = frame.shape[0]
    width = frame.shape[1]
    exp_image = frame.copy()
    red = (970,542)
    green = (0,height)
    blue = (1792,height)
    white = (1129,542)
    cv2.circle(exp_image, red, radius=5, color=(255, 0, 0), thickness=10)     ##RED
    #cv2.circle(exp_image, (10,530), radius=5, color=(0, 0, 255), thickness=10)     

    cv2.circle(exp_image, green, radius=5, color=(0, 255, 0), thickness=10)     ##GREEN
    cv2.circle(exp_image, blue, radius=5, color=(0, 0, 255), thickness=10)    ##BLUE
    cv2.circle(exp_image, white, radius=5, color=(255, 255, 255), thickness=10)       ##WHITE
        #break
    #print(exp_image)
    #fig = plt.figure()
    #fig.add_subplot(111)
    #plt.imshow(exp_image)
    
    
    mask = np.zeros_like(frame) #create an array of 0 of size image i.e no. of pixels will be equal
    print(type(mask))
    cv2.fillPoly(mask,np.array([[list(red), list(green), list(blue), list(white)]]),255) #here we are merging mask and polygon and we color of the polygon will be 255(white)
    
    masked_image = cv2.bitwise_and(frame,mask)
    #fig.add_subplot(121)
    #plt.imshow(masked_image)
    
    
    
    
    
    
    
    figure, (ax1, ax2) = plt.subplots(2,1) # 2 rows, 1 column
    figure.set_size_inches(10, 5)
    ax1.imshow(exp_image, cmap='gray')
    ax1.set_title("Warped Binary Frame")
    ax2.imshow(masked_image)
    ax2.set_title("Histogram Peaks")
    plt.show()

    
    
    
    return np.float32([red, green, blue, white])




#if __name__ == "__main__" :

region_of_interest_pts = find_roi(0)
print(region_of_interest_pts)

with open('RoiPoints.npy', 'wb') as f:
    np.save(f, region_of_interest_pts)
with open('RoiPoints.npy', 'rb') as f:
    a = np.load(f)
    print(a)



    
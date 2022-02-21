import cv2
import numpy as np
import matplotlib.pyplot as plt
from RoiUtils import get_roi


def canny(image):
    #to create gray scale image
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    canny = cv2.Canny(blur,50,150) #sobel transform
    return canny



#imshape = lane_image.shape
def perspective_transformation(output_first,region_of_interest_pts):
    #region_of_interest_pts = np.float32([(520,300),(290,height),(1010,height),(600,300)]) **
    #region_of_interest_pts = np.array([[(0,imshape[0]),(imshape[1]*.48, imshape[0]*.58), (imshape[1]*.52, imshape[0]*.58), (imshape[1],imshape[0])]], dtype=np.float32)              

    #desired_region_of_interest_pts = np.float32([(0,0),(height,0),(width,height),(width,0)])
    desired_region_of_interest_pts = np.float32([[0, 0],[0, height],[width, height],[width, 0]])
    # Calculate the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(region_of_interest_pts,desired_region_of_interest_pts)
    # Calculate the inverse transformation matrix
    inv_transformation_matrix = cv2.getPerspectiveTransform(desired_region_of_interest_pts,region_of_interest_pts)
    warped_image = cv2.warpPerspective(output_first,transformation_matrix,(width,height),flags=(cv2.INTER_LINEAR))
    warped_image_copy = warped_image.copy()
    warped_plot = cv2.polylines(warped_image_copy, [desired_region_of_interest_pts.astype("int32")], True, (255,255,255), 50)
    #cv2.polylines(exp , [region_of_interest_pts] , True , (255,255,255)  , 5)
    #plt.imshow(warped_image)
    return warped_image,inv_transformation_matrix



#here we get the left and right peak of histogram
def histogram_peak_indices(histogram):
    #print(histogram.shape)
    midpoint = np.int(histogram.shape[0]/2)
    leftx = np.argmax(histogram[0:midpoint])
    rightx = np.argmax(histogram[midpoint:]) + midpoint
    return leftx,rightx



def get_line_indices_sliding_windows(warped_image_binary):
    frame_sliding_window = warped_image_binary.copy()
    
    #to set height of the sliding window
    window_height = np.int(warped_image_binary.shape[0]/no_of_windows)
    
    #to find the nonzero(white) pixels in the frame
    nonzero = warped_image_binary.nonzero() #returns indices with nonzero elements
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    #print(nonzerox,"nonzerox")
    #to store the pixel indices for left and right lane lines
    left_lane_inds = []
    right_lane_inds = []
    
    #current position for pixel indices for each window which we will update conti.
    leftx_base, rightx_base = histogram_peak_indices(histogram)
    leftx_current = leftx_base
    rightx_current = rightx_base
   
    for window in range(no_of_windows):
        
        #identify window boundaries in x and y
        win_y_low = warped_image_binary.shape[0] - (window + 1) * window_height
        win_y_high = warped_image_binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        #in cv2.rectancle 1st tuple is top-left corner coordinates & 2nd tuple is bottom-right coordinates
        #this 1st is for the left lane
        cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (win_xleft_high,win_y_high), (255,255,255), thickness=2)
        #this is for the right lane
        cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255,255,255), 2)
        
        #identify the x and y of nonzero pixels within the window
        #nonzero returns a tuple but we want only the array of nonzero indices so we did nonzero()[0] **
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        #print(good_right_inds,"good_right_inds")
        #append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        #if you found > minpix(minimun no.of pixels) pixels, recenter next window on mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    #print(right_lane_inds,"right_lane_inds")
    #concatenate array of indices
    #np.concatenate concatenates all the arrays to create one single array
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    #extract the pixel coordinates from left and right lane lines
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds] 
    righty = nonzeroy[right_lane_inds]
    
    #fit a 2 degree polynomial curve to the pixel coordinates
    #the left and right lane lines
    #print(rightx,righty,"rightx,righty")
    #print(lefty,leftx,"lefty,leftx")
    left_fit = np.polyfit(lefty, leftx, deg = 2)
    right_fit = np.polyfit(righty, rightx,deg= 2)

    #create the x and y values to plot on the image
    ploty = np.linspace(0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2] 

    #visualization
    #print(frame_sliding_window.shape)
    out_img = np.dstack((frame_sliding_window, frame_sliding_window, (frame_sliding_window))) * 255

    #adding color to left line pixels and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = (255,0,0)
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = (0,0,255)
    
    '''
    figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
    figure.set_size_inches(10, 10)
    figure.tight_layout(pad=3.0)
    ax1.imshow(cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
    ax2.imshow(frame_sliding_window, cmap='gray')
    ax3.imshow(out_img)
    ax3.plot(left_fitx, ploty, color='yellow')
    ax3.plot(right_fitx, ploty, color='yellow')
    ax1.set_title("Original Frame")  
    ax2.set_title("Warped Frame with Sliding Windows")
    ax3.set_title("Detected Lane Lines with Sliding Windows")
    #plt.show()
    
    #plt.imshow(out_img)
    '''
    #plt.imshow(frame_sliding_window)
    return left_fit, right_fit




def get_lane_line_previous_window(warped_image_binary,left_fit, right_fit):
    
    #to find the nonzero(white) pixels in the frame
    nonzero = warped_image_binary.nonzero() #returns indices with nonzero elements
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
         
    # Store left and right lane pixel indices
    left_lane_inds = ((nonzerox > (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
      nonzerox < (left_fit[0]*(
      nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
      nonzerox < (right_fit[0]*(
      nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))           
 
    # Get the left and right lane line pixel locations  
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]      
     
    # Fit a second order polynomial curve to each lane line
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
         
    # Create the x and y values to plot on the image
    ploty = np.linspace(
      0, warped_image_binary.shape[0]-1, warped_image_binary.shape[0]) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
         
    # Generate images to draw on
    out_img = np.dstack((warped_image_binary, warped_image_binary, (warped_image_binary)))*255
    window_img = np.zeros_like(out_img)

    # Add color to the left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Create a polygon to show the search window area, and recast 
    # the x and y points into a usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([
                                 right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                 right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the figures 
    '''
    figure, (ax1, ax2, ax3) = plt.subplots(3,1) # 3 rows, 1 column
    figure.set_size_inches(10, 10)
    figure.tight_layout(pad=3.0)
    ax1.imshow(cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
    ax2.imshow(warped_image_binary, cmap='gray')
    ax3.imshow(result)
    ax3.plot(left_fitx, ploty, color='yellow')
    ax3.plot(right_fitx, ploty, color='yellow')
    ax1.set_title("Original Frame")  
    ax2.set_title("Warped Frame")
    ax3.set_title("Warped Frame With Search Window")
    plt.show()
    '''
    
    return left_fitx,right_fitx,ploty,leftx,lefty,rightx,righty




def radius_of_curvature_and_distance(ploty,leftx,lefty,rightx,righty):
    
    y_eval = np.max(ploty)
    
    left_fit_curve = np.polyfit(lefty*ymeter_per_pixel,leftx*xmeter_per_pixel,2)
    right_fit_curve = np.polyfit(righty*ymeter_per_pixel,rightx*xmeter_per_pixel,2)
    
    #calculate the radii of curvature
    left_curvature = ((1 + (2*left_fit_curve[0]*y_eval*ymeter_per_pixel + left_fit_curve[1])**2)**1.5) / np.absolute(2*left_fit_curve[0])
    right_curvature = ((1 + (2*right_fit_curve[0]*y_eval*ymeter_per_pixel + right_fit_curve[1])**2)**1.5) / np.absolute(2*right_fit_curve[0])
    
    return left_curvature, right_curvature




def overlay_lane_lines(frame, warped_image_binary,left_fitx,right_fitx,ploty,inv_transformation_matrix):
    #print(warped_image_binary.shape,"warped_image_binary.shape")
    #plt.imshow(warped_image_binary)
    # Generate an image to draw the lane lines on 
    warp_zero = np.zeros_like(warped_image_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))       
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([
                            left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([
                            right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective 
    # matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inv_transformation_matrix, (
                                    frame.shape[
                                    1], frame.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    ##########################################
    '''
    # Create the x and y values to plot on the image
    ploty = np.linspace(
      0, warped_image_binary.shape[0]-1, warped_image_binary.shape[0]) 
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
         
    # Draw lane on the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
 
    # Warp the blank back to original image space using inverse perspective 
    # matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, inv_transformation_matrix, (
                                  lane_image.shape[
                                  1], lane_image.shape[0]))
     
    # Combine the result with the original image
    result = cv2.addWeighted(lane_image, 1, newwarp, 0.3, 0)
      
    # Plot the figures 
    
    figure, (ax1, ax2) = plt.subplots(2,1) # 2 rows, 1 column
    figure.set_size_inches(10, 10)
    figure.tight_layout(pad=3.0)
    ax1.imshow(cv2.cvtColor(lane_image, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Frame")  
    ax2.set_title("Original Frame With Lane Overlay")
    plt.show()   
    
    '''
    #plt.imshow(result)
    return result





def display_on_screen(image,left_curvature, right_curvature):
    image_copy = image.copy()
    cv2.putText(image_copy,'Curve Radius: '+str((
      left_curvature+right_curvature)/2)[:7]+' m', (int((
      5/600)*width), int((
      20/338)*height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
      0.5/600)*width)),(
      255,255,255),2,cv2.LINE_AA)
    plt.imshow(image_copy)
    return image_copy







import cv2
import numpy as np
import matplotlib.pyplot as plt
#from RoiUtils import get_first_frame
#now we have to do this for each frame in video
#Highway - 10364.mp4

road_video = cv2.VideoCapture("Lane Detection/Highway - 10364.mp4")
i=0

#width = frame.shape[1]
#height = frame.shape[0]
width = int(road_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(road_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Sliding window parameters
no_of_windows = 10
margin = int((1/12) * width)  # Window width is +/- margin
minpix = int((1/24) * width)

# Pixel parameters for x and y dimensions
ymeter_per_pixel = 10.0 / 1000 # meters per pixel in y dimension
xmeter_per_pixel = 3.7 / 781 # meters per pixel in x dimension

#region_of_interest_pts = find_roi("Highway - 10364.mp4")
region_of_interest_pts = get_roi()

desired_region_of_interest_pts = np.float32([[0, 0],[0, height],[width, height],[width, 0]])





while(road_video.isOpened()):
    flag = True
    _,frame = road_video.read() #replace lane_image with frame
    if _ == True:
        try :
            #frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_AREA)
        
        #cv2.imshow("frame", frame)
        
            '''
            canny_image = canny(frame)
            #cro
            #cropped_image = canny_image
            cropped_image = region_of_interest(canny_image)
            lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength= 40, maxLineGap=5)
            averaged_lines = average_slope_intercept(frame,lines)
            line_image = display_lines(frame,averaged_lines)
            combo_image = cv2.addWeighted(frame, 0.8, line_image , 1,1)
            cv2.imshow("Frame", combo_image)


            
            #output_first = combo_image.copy()
            
            
            

            # Resize the frame

            # Perform the perspective transform to generate a bird's eye view
            '''
            canny_image2 = canny(frame)
            warped_image, inv_transformation_matrix = perspective_transformation(canny_image2,region_of_interest_pts)
            #warped_image = warped_image[0]

            # Generate the image histogram to serve as a starting point
            # for finding lane line pixels

            threshold_value,warped_image_binary = cv2.threshold(warped_image, 127, 255, cv2.THRESH_BINARY)
            #cv2.imshow("frame", warped_image_binary)


            histogram = np.sum(warped_image_binary[warped_image_binary.shape[0]//2:,:], axis=0)

            # Find lane line pixels using the sliding window method 
            #print(i)
            if i == 0:
                left_fit, right_fit = get_line_indices_sliding_windows(warped_image_binary)
            # Fill in the lane line
            left_fitx,right_fitx,ploty,leftx,lefty,rightx,righty = get_lane_line_previous_window(warped_image_binary,left_fit, right_fit)
            #result1 = result1[:,:]
            
            #to find radii of curvature
            left_curvature, right_curvature = radius_of_curvature_and_distance(ploty,leftx,lefty,rightx,righty)
            
            # Overlay lines on the original frame
            result = overlay_lane_lines(frame, warped_image_binary,left_fitx,right_fitx,ploty,inv_transformation_matrix)
            
            frame_with_lane_lines = display_on_screen(result,left_curvature, right_curvature)

            # Display the frame 
            
            
            cv2.imshow("Frame", frame_with_lane_lines) 
            #plt.imshow(frame_with_lane_lines)
            # Display frame for X milliseconds and check if q key is pressed
            # q == quit
            #return frame_with_lane_lines
            
            
        except :
            
            cv2.imshow("Frame", frame)
            print("not found")

        if cv2.waitKey(1) == ord('q'):
            break
        
    # No more video frames left
    else:
        break
    i+=1
            
# Stop when the video is finished
road_video.release()

# Release the video recording
road_video.release()

# Close all windows
cv2.destroyAllWindows()
print("hi")



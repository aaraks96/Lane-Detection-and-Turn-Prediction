import numpy as np
import cv2
from matplotlib import pyplot as plt
name = "project_video.mp4"
cap = cv2.VideoCapture(name)
def correct_dist(initial_img):
    k = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    k = np.array(k)
    # Distortion Matrix
    dist = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
    dist = np.array(dist)
    img_2 = cv2.undistort(initial_img, k, dist, None, k)

    return img_2
def curvature(y, p):
    return ((1 + (2 * p[0] * y + p[1])**2)**(1.5)) / np.absolute(2 * p[0])  
    
def polynomial_filling(left_x, left_y, right_x, right_y, edged):
    y_scaling = 30/720
    x_scaling = 3.7/700
    
    poly_left = np.polyfit(left_y, left_x,2)
    poly_right = np.polyfit(right_y, right_x,2)
    y_vals = np.linspace(0, edged.shape[0]-1, edged.shape[0])
    y_vals_max = np.max(y_vals)
    poly_left_x = poly_left[0]*y_vals**2 + poly_left[1]*y_vals + poly_left[2]
    roc_left = ((1 + (2 * poly_left[0] * left_y + poly_left[1])**2)**(1.5)) / np.absolute(2 * poly_left[0])
    poly_right_x = poly_right[0]*y_vals**2 + poly_right[1]*y_vals + poly_right[2]
    roc_right = ((1 + (2 * poly_right[0] * right_y + poly_right[1])**2)**(1.5)) / np.absolute(2 * poly_right[0])
    
    left_lane0 = np.array([np.transpose(np.vstack([poly_left_x, y_vals]))])
    right_lane0 = np.array([np.transpose(np.vstack([poly_right_x, y_vals]))])
    
   
    poly_left_new = np.polyfit(left_y*y_scaling, left_x*x_scaling,2)
    poly_right_new = np.polyfit(right_y*y_scaling, right_x*x_scaling,2)
    
    roc_left = ((1 + (2 * poly_left_new[0] * y_vals_max*y_scaling + poly_left_new[1])**2)**(1.5)) / np.absolute(2 * poly_left_new[0])
    roc_right = ((1 + (2 * poly_right_new[0] * y_vals_max*y_scaling + poly_right_new[1])**2)**(1.5)) / np.absolute(2 * poly_right_new[0])
    left_line_points1 = left_lane0.reshape((left_lane0.shape[1],-1))
    right_line_points1 = np.flipud(right_lane0.reshape((right_lane0.shape[1],-1)))
    
    total_lane = np.array(np.concatenate((left_line_points1,right_line_points1)),dtype = 'int32')
    
    
    blank_background = np.zeros_like(edged)
    cv2.fillPoly(blank_background, [total_lane], 255)
    #cv2.fillPoly(blank_background, np.int_([right_line_points]), 255)
    
    
    
    return blank_background, left_line_points1, right_line_points1, roc_left, roc_right
    
        
def lane_detection(edged,cropped_image):
    windows = 10
    size = 80

    edge_height = edged.shape[0]
    edge_width = edged.shape[1]
    
    win_height = edge_height//windows
    cropped_height = cropped_image.shape[1]
    cropped_width = cropped_image.shape[0]
    histogram = np.sum(edged, axis=0)
    
    mid_pt = histogram.shape[0]//2
    window_left = np.argmax(histogram[:mid_pt])
    window_right = np.argmax(histogram[mid_pt:])+mid_pt
    pixel_locations =  edged.nonzero()
    pixel_locations_x = np.array(pixel_locations[1])
    pixel_locations_y = np.array(pixel_locations[0])
    
    left_pixel_list = []
    right_pixel_list = []
    
    for window in range(windows):
        
        win_bottom = edge_height - (win_height*window)
        win_top = edge_height - (win_height*(window+1))
        win_left_bottom = window_left - size
        win_left_top = window_left + size
        win_right_bottom = window_right - size
        win_right_top = window_right + size
        
        
        
        left_pixels = ((pixel_locations_y>=win_top)&(pixel_locations_y<win_bottom)
                        &(pixel_locations_x>=win_left_bottom)& (pixel_locations_x<win_left_top)).nonzero()[0]
        
        right_pixels = ((pixel_locations_y>=win_top)&(pixel_locations_y<win_bottom)
                        &(pixel_locations_x>=win_right_bottom)& (pixel_locations_x<win_right_top)).nonzero()[0]
        
        left_pixel_list.append(left_pixels)
        right_pixel_list.append(right_pixels)

    left_pixel_list = np.concatenate(left_pixel_list)
    right_pixel_list = np.concatenate(right_pixel_list)
    left_x = pixel_locations_x[left_pixel_list]
    left_y = pixel_locations_y[left_pixel_list]
    right_x = pixel_locations_x[right_pixel_list]
    right_y = pixel_locations_y[right_pixel_list] 
    if left_x.size==0 & left_y.size==0 & right_x.size == 0 & right_y.size ==0:
        return [0],[0],[0],[0],[0]
    
    temp_image = np.dstack((edged,edged,edged))*255
    blank = np.zeros_like(temp_image)
    temp_image[pixel_locations_y[left_pixel_list], pixel_locations_x[left_pixel_list]] = [0,255,0]
    temp_image[pixel_locations_y[right_pixel_list], pixel_locations_x[right_pixel_list]] = [0,0,255]
    
    
    
    return temp_image, left_x,left_y, right_x, right_y



while (cap.isOpened()):
    success, frame = cap.read()
    if success == False:
        break
    
    frame = correct_dist(frame)
    frame_size = frame.shape    
    cropped_image = frame[420:720,40:1280,:]
    
    frame_hsl = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2HLS)
    
    masked_image = np.zeros(frame_hsl.shape)
    yellow_upper_range = np.array([45,200,255], dtype = 'uint8')
    yellow_lower_range = np.array([20,10,80], dtype = 'uint8')
    yellow_mask = cv2.inRange(frame_hsl,yellow_lower_range,yellow_upper_range)
    white_lower_range = np.array([0,200,0], dtype = 'uint8')
    white_upper_range = np.array([255,255,255], dtype = 'uint8')
    
    white_mask = cv2.inRange(frame_hsl, white_lower_range, white_upper_range)
    
    mask_combined = cv2.bitwise_or(yellow_mask, white_mask)
       
    masked_image[:,:,0] = cv2.bitwise_and(frame_hsl[:,:,0], mask_combined).astype(np.uint8)
    masked_image[:,:,1] = cv2.bitwise_and(frame_hsl[:,:,1], mask_combined).astype(np.uint8)
    masked_image[:,:,2] = cv2.bitwise_and(frame_hsl[:,:,2], mask_combined).astype(np.uint8)
    masked_image = np.array(masked_image,dtype = 'uint8')
    
    mat_h = np.array([[-2.50638675e+00, -4.84647803e+00, 1.44429332e+03], [-1.24287961e+00, -2.34766149e+01, 1.81515663e+03], [-2.19653696e-03, -3.38621948e-02, 1.00000000e+00]])
    
    bil = cv2.bilateralFilter(mask_combined,9,120,100)
    # sobelx = np.uint8(cv2.Sobel(mask_combined,cv2.CV_64F,1,0,ksize=5))
    # edge = cv2.Canny(mask_combined, 120,200)
    new_img = cv2.warpPerspective(bil, mat_h, (300, 600))
    #cv2.imshow("warped",new_img)   
    src_pt = np.array([[50, 0], [250, 0], [250, 550], [50, 550]], dtype="float32")
    dst_pt = np.array([[516, 50], [686, 41], [1078, 253], [241, 259]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pt, dst_pt) 
    frame_win = cropped_image.copy()
    filled, left_x, left_y, right_x, right_y = lane_detection(new_img,frame_win)
    if np.array(left_x).all() ==0:
        continue
    background, left_points, right_points,roc_left,roc_right = polynomial_filling(left_x, left_y,right_x,right_y,filled)
    #cv2.imshow("edged", background)
    filled = cv2.warpPerspective(filled, M, (cropped_image.shape[1],cropped_image.shape[0]))
    #cv2.imshow("filled",filled)
    
    new_img2 = cv2.warpPerspective(background, M, (cropped_image.shape[1],cropped_image.shape[0]))
    
    lane_color = cv2.addWeighted(filled,2,new_img2,0.4,0)
    
    final_image = cv2.bitwise_or(lane_color,cropped_image)
     
    #print("roc_left,roc_right",(roc_left,roc_right))
    if(roc_left<=3000 and  roc_right<=3000):
        if(roc_left>roc_right):
            text = "right"
        else:
            text = "left"
    
    else:
            text = " "
    cv2.putText(final_image,text,(50,100), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),2,cv2.LINE_AA)
    cv2.imshow("Final Frame",final_image) 
    #cv2.waitKey(0)
    if cv2.waitKey(23) & 0xff == 27:
        cv2.destroyAllWindows()
    
cap.release()



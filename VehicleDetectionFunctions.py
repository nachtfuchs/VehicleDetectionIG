''' @brief  This contains relevant functions for the project assignments for the Udacity 
            CarND-Project 5 "Vehicle Detection"
'''

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #only valid for scikit-learn version >= 0.18
from scipy.ndimage.measurements import label
# from CVehicle import CVehicle
import CVehicle #debug
########################
# function definitions
#########################

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis = False, feature_vec = True):
    ''' @brief This function returns the Histogram of Oriented Gradient features for a particular input image
                Note: Contains two outputs, if vis = True
    '''
    
    if vis ==  True:
        features, hog_image = hog(img, orientations = orient,
                                  pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cell_per_block, cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis,
                                  feature_vector = feature_vec)
        return features, hog_image
    else:
        features = hog(img, orientations = orient,
                                  pixels_per_cell = (pix_per_cell, pix_per_cell),
                                  cells_per_block = (cell_per_block, cell_per_block),
                                  transform_sqrt = False,
                                  visualise = vis,
                                  feature_vector = feature_vec)
        return features
    
def bin_spatial(img, color_space='RGB', size=(32, 32)):
    ''' @brief Downsampling an input image into the desired size.
    '''
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)             
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(feature_image, size).ravel() 
    # Return the feature vector
    return features

def color_hist(img, nbins=32): 
    ''' @brief Returns the histogram of the image for all three (!)color channels.
    '''
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(img[:,:,0], bins=nbins)
    ghist = np.histogram(img[:,:,1], bins=nbins)
    bhist = np.histogram(img[:,:,2], bins=nbins)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the feature vector
    return hist_features

def color_hist_sub(img, nbins=32):    #bins_range=(0, 256)
    ''' @brief  Returns the histogram of the image for all three (!)color channels for 
                the sub-sampling approach
    '''
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
#     hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    hist_features = np.hstack((channel1_hist[0], channel2_hist[0], channel3_hist[0])) #debug
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    ''' @brief Used for training a classifier
    '''
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        # Apply color_hist() also with a color space option now
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        # Append the new feature vector to the features list
        
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        #features.append(np.concatenate((spatial_features, hist_features, hog_features)))
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis = False):
    ''' @brief 
    '''   
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis = vis, feature_vec=True))      
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis = True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis = False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    if vis == True and hog_channel != 'ALL':
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, x_start, x_stop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step = 2, window = 64):
    ''' @brief This function performs the HOG on the whole picture instead of each window.
                The benefit is that it should perform much faster than the approach to define windows
                in the image, and then search for hog features within the individual windows.
                CarND-lesson "Hog Sub-sampling Window Search". 
    '''
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255 # for jpg files
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float) # cold heatmap
    img_boxes = [] # used for storing image boxes
    img_tosearch = img[ystart:ystop,:,:]
    img_tosearch = img_tosearch[:, x_start:x_stop, :] # the left plane is not of high importance in this project. 
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = window
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 1  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (window, window))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist_sub(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                img_boxes.append( ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart) ))
                heatmap[ ytop_draw + ystart : ytop_draw + win_draw + ystart, xbox_left : xbox_left + win_draw] += 1 
                
    return draw_img, heatmap
    
def add_heat(heatmap, bbox_list):
    ''' @brief  Add heat to a canvas "heatmap" where the input bounding boxes cover area
    @input heatmap A numpy matrix of the size of the image under analysis with one color channel
    @input bbox_list A list of bounding boxes where an object of interest was identified
    '''
    
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    ''' @brief Apply a threshold for a heatmap to get rid of false positives
    @input heatmap An input numpy matrix where pixels with values larger than
                    zero are considered to be an object of interest
    @input threshold An integer number that should neglect false positives
    '''
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, x_start):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox)+ x_start, np.min(nonzeroy)), (np.max(nonzerox) + x_start, np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def visualize(fig, rows, cols, imgs, titles):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(i+1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap = 'hot')
            plt.title(titles[i])
        else:
            plt.imshow(img)
            plt.title(titles[i])
    plt.show()

def calc_overlap(labels, car_list):
    ''' @brief Calculate the overlap between the given labels and the detected vehicle boxes. The idea
                is that the largest overlap represents the car from the past within the newest image. 
    @input labels     A return value of the "scipy.ndimage.measurements import label()"-function
    @input car_list   A list of vehicle objects of the CVehicle class
    '''
    # Deal with empty labels
    if labels[1] == 0:
        return None
    score = [] # a list of scores -> number of overlapping pixels.
#     if labels[1] > 1: #debug
#         print('calc_overlap: more than one car in labels.') #debug
    # go through all labels
    for i in range(1, labels[1] + 1):
        # go through all cars
        for car in car_list:
            # determine the pixels where the label class is found
            nonzero_label = (labels[0] == i).nonzero() #contains the x and y pixel where the label is found
            label_x = nonzero_label[1][:]
            label_y = nonzero_label[0][:]
            # determine the car box
            car_box_x = car.xpixels
            car_box_y = car.ypixels
            # count how many pixels overlap using sets
            car_box_set = set( zip( car_box_y, car_box_x) )
            label_set = set( zip( label_y, label_x) )
            intersect = car_box_set & label_set # pairs of equal points
            # count the equal points
            score.append( len(intersect) )
    # reshape into matrix structure where the row number equals the label number and the column equals the car number
    score_mat = np.array(score).reshape(labels[1], len(car_list))
    return score_mat

def merge_cars(car_list, overlap_thresh): #debug
    ''' @brief  This function merges car objects that have a high overlap of their window
    @input car_list A list of CVehicle objects 
    @input overlap_thresh A float number between 0 and 1 that represents the overlap threshold.
                          If the overlap is larger than the overlap, cars will be merged.
    '''
    a = 0 # loop variable
    b = 0 # loop variable
    overlap_pair = [] # list of overlap pairs 
    for car_a in car_list:
        for car_b in car_list:
            if b >= a:
                if car_a != car_b:
                    # calculate overlap
                    car_a_box_x = list( range( car_a.bestx, car_a.bestx +  car_a.bestw))
                    car_a_box_y = list( range( car_a.besty, car_a.besty +  car_a.besth))
                    
                    car_b_box_x = list( range( car_b.bestx, car_b.bestx +  car_b.bestw))
                    car_b_box_y = list( range( car_b.besty, car_b.besty +  car_b.besth))
                    
                    car_a_set = set( zip( car_a_box_x, car_a_box_y) )
                    car_b_set = set( zip( car_b_box_x, car_b_box_y) )
                    # calculate overlapping points
                    intersect = car_a_set & car_b_set
                    # determine which box is smaller
                    if len(car_a_box_x) <= len(car_b_box_x):
                        ref = len(car_a_box_x) # take the smaller box as a reference
                    else:
                        ref = len(car_b_box_x)
                    
                    # determine if merging is necessary
                    if len(intersect) >= (overlap_thresh * ref):
                        overlap_pair.append((a, b))
                b += 1
            else:
                b += 1
        a += 1
        b = 0
    # make list unique 
#     if len(overlap_pair) > 0: #debug
#         print('merge_cars: overlap_pair > 0') #debug
   
    for pair in overlap_pair:
        # determine the larger box
        box_a = car_list[pair[0]].bestw + car_list[pair[0]].besth # box size is represented by the length of its two sides
        box_b = car_list[pair[1]].bestw + car_list[pair[1]].besth
        if box_a >= box_b:
            print('car_list[pair[1]]: ', car_list[pair[1]])
#             merge_car_properties(car_list[pair[0]], car_list[pair[1]])
            car_list[pair[0]].merge(car_list[pair[1]])
            del car_list[pair[1]] # delete the merged object
        else:
            print('car_list[pair[0]]: ', car_list[pair[0]])
#             merge_car_properties(car_list[pair[1]], car_list[pair[0]])
            car_list[pair[1]].merge(car_list[pair[0]]) 
            del car_list[pair[0]] # delete the merged object
#         print('merge_cars: objects merged') # debug
         
    
def analyze_overlap_matrix(mat, labels, car_list):
    ''' @brief Analyze the overlap matrix and assign labels to detected cars.
    @input mat The return value of the calc_overlap function. The row number equals the label number and the column equals the car number
    '''
#     if labels[1] > 1: #debug
#         print('analyze_overlap_matrix: more than one car in labels.') #debug
    # Deal with unidentified labels
    if mat == None:
        # set the non detected properties of each car
        for car in car_list:
            car.detected = False
            car.n_nondetections += 1
    else:
        # search for zero columns
        i = 1
        used_label = [] # contains used labels
        used_cars = [] # contains the indices of cars that received a label -> helps to keep track which cars were identfied in the current image
        for row in mat:
            if np.sum(row) == 0:
                # then you have a new vehicle candidate
                new = CVehicle.CVehicle()
                new.new_detection(labels, i)
                car_list.append(new)
                used_label.append( i)
                used_cars.append( len(car_list) - 1 )
            else:
                # then you search for the index (= car) of the maximum value
                ind = np.argmax(row)
                # then you assign that label to the car
                car_list[ind].add_box(labels, i)
                used_label.append( i)
                used_cars.append( ind ) 
                
            i += 1 #increment the label running variable
        # examine some features of the cars
        for i in range(0, len(car_list)):
            # check if there are cars that were not assigned to labels
            if i not in used_cars:
                car_list[i].not_detected()
        
        # delete the object if it was not detected in certain number of previous images in a row.
        for i in range(len(car_list) - 1, 0, -1):
            if car_list[i].n_nondetections >= 3:
                del car_list[i]
        
        # seperate loop, because objects might be deleted in upper loop
        for i in range(0, len(car_list)): 
            # calculate the best fits
            car_list[i].calc_best()
            
        
def get_car_bbox(car_list, x_start):
    ''' @brief  Get all bounding boxes from the input list of CVehicle objects.
    @input car_list A list of CVehicle objects
    @input x_start  The offset in x-direction that was used to reduce the search area.
                    Needs to be added to the bounding boxes 
    '''
#     if len(car_list) > 1: #debug
#         print('get_car_bbox: more than one car in car_list.') #debug
        
    bbox = [] # list of bounding boxes
    if len(car_list) == 0: # handle empty boxes
        print('No car objects identified.')
        return bbox
    else:
        for car in car_list:
            # draw boxes of cars that were detected more often than a particular amount of times
            if car.n_detections >= 3:
                car_box_min = (car.bestx + x_start, car.besty)                            # top left corner of the box
                car_box_max = (car.bestx + car.bestw + x_start, car.besty + car.besth)    # lower right corner of the box
                bbox.append((car_box_min, car_box_max))
#             else:
# #                 print('get_car_bbox: not enough detections') #debug
#                 dummy = 0
        return bbox

def init_car_list(labels, car_list):
    ''' @brief Initialize a list of CVehicle objects 
    @input labels     A return value of the "scipy.ndimage.measurements import label()"-function
    @input car_list   A list of vehicle objects of the CVehicle class
    '''
    
    if len(car_list) == 0:
        if labels[1] > 0:
            # initialize a car for every label found
            for label in range(1, labels[1]+ 1):
                new_car = CVehicle.CVehicle()
                new_car.new_detection(labels, label) # get the first values 
                car_list.append(new_car)
        else:
            print('No labels found')
    return car_list

def merge_car_properties(merged_car, car_to_merge):
    ''' @brief This function merges the properties of the two input vehicles.
    @merged_car CVehicle This object will inherit the properties of the second input
    @car_to_merge CVehicle This object will pass its properties to the first input.
                           car_to_merge will be deleted.
    '''
    merged_car.n_detections = 0
    merged_car.n_nondetections = 0
    # assign the best values of the second object to the self object
    merged_car.recent_xfitted.append(car_to_merge.bestx) 
    merged_car.recent_yfitted.append(car_to_merge.besty)
    merged_car.recent_hfitted.append(car_to_merge.besth)
    merged_car.recent_wfitted.append(car_to_merge.bestw)
    # calculate the new best fit
    merged_car.calc_best()

def disp_vehicle_info(car_list, draw_img):
    ''' @brief The goal of this function is to write vehicle properties of the input list objects into the input image
    @input car_list A list of CVehicle objects
    @input draw_img A input image that already contains the bounding boxes of particular vehicles.
    '''
    i = 1 # necessary to display the vehicle information correctly
    if len(car_list) > 0: #avoid dealing with empty lists
        for car in car_list:
            font = cv2.FONT_HERSHEY_SIMPLEX
            draw_img = cv2.putText(draw_img, 'CVehicle: ' + str(car), (5,20 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_img = cv2.putText(draw_img, 'detected: ' + str(car.detected), (5, 40 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_img = cv2.putText(draw_img, 'n_detections: ' + str(car.n_detections), (5, 60 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_img = cv2.putText(draw_img, 'n_NONdetections: ' + str(car.n_nondetections), (5, 80 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_img = cv2.putText(draw_img, 'bestx: ' + str(car.bestx), (5, 100 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_img = cv2.putText(draw_img, 'besty: ' + str(car.besty), (5, 120 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_img = cv2.putText(draw_img, 'bestw: ' + str(car.bestw), (5, 140 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            draw_img = cv2.putText(draw_img, 'besth: ' + str(car.besth), (5, 160 + 160 * (i - 1)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            i += 1
    return draw_img

# def calc_heatmap_cog(labels):
#     ''' @brief The function calculates the center of gravity (cog) for the labels of a heatmap
#     @input labels A return value of the "scipy.ndimage.measurements import label()"-function
#     '''
#     # check if any labels were identfied using the number of found labels
#     if labels[1] == 0:
#         print('calc_heatmap_cog: No labels identfied.')
#         return None
#     cog = [] # a list of tupels where the first element of the tupel is the x pixel and the second 
#              # element is the y pixel of the center of gravity
#     # go through the number of labels and calculate their center of gravity
#     for label_number in range(1, labels[1] + 1):
#         # identify the label number
#         nonzero = (labels[0] == label_number).nonzero()
#         # Identify x and y values of those pixels
#         nonzeroy = np.array(nonzero[0])
#         nonzerox = np.array(nonzero[1])
#         p_min = (np.min(nonzerox), np.min(nonzeroy)) # point of upper left corner
#         p_max = (np.max(nonzerox), np.max(nonzeroy)) # point of lower right corner
#         x_cog = p_min[0] + (np.max(nonzerox) - np.min(nonzerox)) // 2 # x pixel of cog
#         y_cog = p_min[1] + (np.max(nonzeroy) - np.min(nonzeroy)) // 2 # y pixel of cog
#         # collect the cogs
#         cog.append((x_cog, y_cog))
#     
#     return cog

# def calc_euclidean_dist(tupel_1, tupel_2):
#     ''' @brief This function calculates the euclidean distance of two points
#     @input tupel_1 A vector input of values (numbers)
#     @output tupel_2 Another vector input of values (numbers).
#     NOTE: Both tupel should have the same axis at identical indices.
#     '''
#     #vectorize the operation
#     a = np.array(tupel_1)
#     b = np.array(tupel_2)
#     
#     dist = np.sqrt(np.sum(np.square(a - b))) # euclidean distance
#     return dist
        
    
# def gating(cogs, labels, car_list):
#     ''' @brief This function assigns detected vehicles from the past to the current labels centers of gravity (cogs)
#     @input cogs A return value from calc_heatmap_cog(). A tupel that represents the cooridnates of the center of gravity for a label.
#     @input labels A return value of the "scipy.ndimage.measurements import label()"-function. Should be the one from the same step as
#                   when the cogs were calculated.
#     @input cars_list A list of vehicle objects of the CVehicle class
#     '''
#     # check if any labels were found
#     if cogs == (None, None):
#         print('gating()-function: cogs is empty.')
#         return None
#     dist = [] # a list of distances
#     # calculate the distance of the label-cogs to the detected vehicles
#     for cog_label_x, cog_label_y in cogs: #loop through the labels
#         for car in car_list: # loop through the detected cars_list
#             car.calc_cog() #calculate the vehicle cog
#             dist = calc_euclidean_dist((cog_label_x, cog_label_y), car.cog)
#     # reshape the list of distances into a matrix of size [rows = number of labels; columns = number of cars]
#     dist_mat = np.array(dist).reshape(len(cogs), len(car_list)) # 'symmetric' matrix if num(labels) = num(cars), except of numerical deviations.
#     # go through the rows (= labels) and find the corresponding minimum column (=car)
#     presumed_car = [] # presumed car stores the car that has the smallest distance to a label(index of the list)
#     for label in dist_mat:
#         # there might be multiple associations one labels and more than one car. And vice versa
#         # search if the current minimum is already in the list of presumed cars
#         if np.argmin(label) in presumed_car:
#             label_subset = np.concatenate( label[:np.argmin(label)], label[np.argmin(label) + 1:])
#             presumed_car.append(np.argmin(label_subset))
#         else: 
#             presumed_car.append(np.argmin(label)) 
        
    # todo: What if there is an unequal number of cars and labels?
    # todo: What if there are no cogs in vehicles or in labels?
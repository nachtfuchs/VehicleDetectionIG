''' @brief  This script is used to solve the project assignments for the Udacity 
            CarND-Project 5 "Vehicle Detection"
'''

import os
import glob
from keras.backend.tensorflow_backend import function
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from VehicleDetectionFunctions import *
from collections import deque

# locate vehicle images
basedir = 'vehicles/'
# Different folders represent different image groups
image_types = os.listdir(basedir)
cars = []
for imtype in image_types:
    cars.extend(glob.glob(basedir + imtype + '/*.png')) #read png's only, because Thumbs.db causes errors
    
print('Number of image vehicles found: ', len(cars))
with open("cars.txt", 'w') as f:
    for fn in cars:
        f.write(fn + '\n')

# Track non-vehicle images
basedir_non_car = 'non-vehicles/'
image_types = os.listdir(basedir_non_car)
notcars = []
for imtype in image_types:
    notcars.extend(glob.glob(basedir_non_car + imtype + '/*.png'))

print('Number of non-vehicles images found: ', len(notcars))
with open("notcars.txt", 'w') as f:
    for fn in notcars:
        f.write(fn + '\n')
        
# choose random car / not-car indices
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# read in car / not-car image
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])
                            
###########################
# define feature parameters
color_space = 'YCrCb' #Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2 or 'ALL'
spatial_bins = (32, 32) # spatial binning dimensions
hist_bins = 64          # number of histogram bins
spatial_feat = True
hist_feat = True
hog_feat = True
###########################
# code commented, once single image features were analyzed -> replaced by extract_features()-function
# car_features, car_hog_image  = single_img_features(car_image, color_space = color_space, spatial_size = spatial_bins,
#                         hist_bins= hist_bins, orient= orient, 
#                         pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel,
#                         spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat, vis = True)
# 
# notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space = color_space, spatial_size = spatial_bins,
#                         hist_bins= hist_bins, orient= orient, 
#                         pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel,
#                         spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat, vis = True)
# 
# images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
# titles =['car_image', 'car_HOG_image', 'notcar_image', 'notcar_HOG_image']
#fig = plt.figure(figsize = (12, 3))
#visualize(fig, 1, 4, images, titles)


############################
# train the SVM classifier 
t = time.time()
n_samples = 1000
random_idxs = np.random.randint(0, len(cars), n_samples)
test_cars = cars # np.array(cars)[random_idxs] # 
test_notcars =  notcars # np.array(notcars)[random_idxs] #     

car_features  = extract_features(test_cars, color_space = color_space, spatial_size = spatial_bins,
                        hist_bins= hist_bins, orient= orient, 
                        pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel,
                        spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat)

notcar_features = extract_features(test_notcars, color_space = color_space, spatial_size = spatial_bins,
                        hist_bins= hist_bins, orient= orient, 
                        pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel,
                        spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat)

print(time.time() - t, ' Seconds to compute features...')
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Fit a per column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)) ))

# split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size = 0.1, random_state = rand_state)

print('Using: ', orient, 'orientations, ', pix_per_cell, 'pixels per cell, ', cell_per_block, 'cells per block, ',
      hist_bins, 'histogram bins, and ', spatial_bins, 'spatial sampling.')
print('Feature vector length: ', len(X_train[0]))
# Use a SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
print(round(time.time() - t, 2), ' Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

##########################
# apply sliding window
search_path = './test_images/*.jpg'
example_images = glob.glob(search_path)
images = []
titles = []
img = mpimg.imread(example_images[0])
y_size = img.shape[0]
y_start_stop = [int(0.5* y_size), 0.9 * y_size] # Min and max values to search in the sliding window function
overlap = 0.2
# The below is commented to reduce compile time during "development" 
# for file in example_images:
#     t1 = time.time()
#     img = mpimg.imread(file)
#     draw_img = np.copy(img) # "canvas" for windows
#     heatmap = np.zeros_like(img[:, :, 0]).astype(np.float) # cold heatmap
#     img = img.astype(np.float32) / 255 # "/255" because of jpg files
#     print(np.min(img, np.max(img ))) # debug
#      
#     # search for different car sizes -> cars in different distances
#     windows_small = slide_window(img, x_start_stop = [None, None], y_start_stop = y_start_stop,
#                            xy_window = [92, 92], xy_overlap = (overlap, overlap))
#     windows_mid = slide_window(img, x_start_stop = [None, None], y_start_stop = y_start_stop,
#                            xy_window = [144, 144], xy_overlap = (overlap, overlap))
#     windows = windows_small + windows_mid #+ windows_large # concatenate all windows 
#     # search for multiple detections and create a heat map
#     hot_windows = search_windows(img, windows, svc, X_scaler, color_space = color_space,
#                                  spatial_size = spatial_bins, hist_bins = hist_bins,
#                                  orient = orient, pix_per_cell = pix_per_cell,
#                                  cell_per_block = cell_per_block,
#                                  hog_channel = hog_channel, spatial_feat = spatial_feat,
#                                  hist_feat = hist_feat, hog_feat = hog_feat)
#      
#     window_img = draw_boxes(draw_img, hot_windows, color = (0, 0, 255), thick = 6)
#     heatmap = add_heat(heatmap, hot_windows)
#     heatmap = apply_threshold(heatmap = heatmap, threshold = 1)
#     images.append(window_img)
#     images.append(heatmap)
#     titles.append('')
#     titles.append('') # twice, because of heatmap
#     print(time.time()- t1, 'seconds to process one image searching', len(windows), 'windows')
# vis_img = draw_boxes(draw_img, windows, color = (0, 0, 255), thick = 6) # for displaying an example sliding window image 
# fig = plt.figure(figsize = (12, 18))
# visualize(fig, 6, 2, images, titles)

##############################
# Perform the car search using HOG sub-sampling

# Parameters for the find_cars()-function
cells_per_step = 2  # "overlap" in cells
window = 64         # window size
ystart = y_start_stop[0]
ystop = y_start_stop[1]
scale_small = 1         # downsampling factor of the image
scale_mid = 1.3
scale_large = 1.7 
images = []
heat_threshold = 3
overlap_thresh = 0.1
# for file in example_images:
#     t1 = time.time()
#     img = mpimg.imread(file)
#       
#     # search different window sizes 
#     out_img_small, out_heatmap_small = find_cars(img = img, ystart = ystart, ystop = ystop, scale = scale_small, svc = svc, X_scaler = X_scaler,\
#                         orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,\
#                         spatial_size = spatial_bins, hist_bins = hist_bins, cells_per_step = cells_per_step, window = window)
#       
#     out_img_mid, out_heatmap_mid = find_cars(img = img, ystart = ystart, ystop = ystop, scale = scale_mid, svc = svc, X_scaler = X_scaler,\
#                         orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,\
#                         spatial_size = spatial_bins, hist_bins = hist_bins, cells_per_step = cells_per_step, window = window)
#       
#     # combine the results of window sizes
#     out_img = out_img_small #(out_img_small + out_img_mid) //2 # -> Todo
#     out_heatmap = out_heatmap_small + out_heatmap_mid
#     out_heatmap = apply_threshold(heatmap = out_heatmap, threshold = heat_threshold)
#      
#     # find labels in heatmap
#     labels = label(out_heatmap)
#      
#     # draw bounding boxes on the "merged" labels
#     draw_img = draw_labeled_bboxes(np.copy(img), labels)
#      
#     images.append(draw_img)
#     images.append(out_heatmap)
#     titles.append('')
#     titles.append('') #for the heatmap
#     print(time.time()- t1, 'seconds to process one image')
#       
# fig = plt.figure(figsize = (12, 18))
# visualize(fig, 2, 6, images, titles)

####################################
# perform car search on video
cap = cv2.VideoCapture('project_video.mp4')
# gather some video information
n_frames = cap.get(7)        # number of total frames in the video
n_frames_counter = n_frames  # counts down the number of processed frames

height , width , layers =  img.shape #used for creating a video
x_start = int(0.5* width)    # ignore most of the left plane to avoid false positives
x_stop = width          # respect all pixels up to the right image border
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
video = cv2.VideoWriter('project_video_my.mp4', fourcc, 18.0,(width,height))

t_video_start = time.time() # measure total video processing time
d = deque(maxlen = 10) # Works as a FIFO
t_remaining = deque(maxlen = 3) # used to estimate the remaining time
 
car_list = [] # stores the CVehicle objects
while (True):
    # measure time
    t_img_start = time.time()
    # Capture frame-by-frame
    ret, img = cap.read()
    if img == None: #end of video
        break # break the loop
    
    dummy, out_heat_small = find_cars(img = img, x_start = x_start, x_stop = x_stop, ystart = ystart, ystop = ystop, scale = scale_small, svc = svc, X_scaler = X_scaler,\
                        orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,\
                        spatial_size = spatial_bins, hist_bins = hist_bins, cells_per_step = cells_per_step, window = window)
    
    dummy, out_heat_mid = find_cars(img = img, x_start = x_start, x_stop = x_stop, ystart = ystart, ystop = ystop, scale = scale_mid, svc = svc, X_scaler = X_scaler,\
                        orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,\
                        spatial_size = spatial_bins, hist_bins = hist_bins, cells_per_step = cells_per_step, window = window)
    
    dummy, out_heat_large = find_cars(img = img, x_start = x_start, x_stop = x_stop, ystart = ystart, ystop = ystop, scale = scale_large, svc = svc, X_scaler = X_scaler,\
                        orient = orient, pix_per_cell = pix_per_cell, cell_per_block = cell_per_block,\
                        spatial_size = spatial_bins, hist_bins = hist_bins, cells_per_step = cells_per_step, window = window)
    # Add the result of all three 
    out_heat = out_heat_small + out_heat_mid + out_heat_large
    
    # Average over the past couple of images
    d.append(out_heat)
    heat_map_array = np.array(d)
    heat_map_average = np.average(heat_map_array, 0)
    out_heat = apply_threshold(heat_map_average, threshold = heat_threshold)
    
    # label on averaged heat map
    labels = label(out_heat)
    # draw boxes
    draw_img = draw_labeled_bboxes(img, labels, x_start)

    # Display the resulting frame
    cv2.imshow('frame', draw_img)
    # debug: save frame to frame to figure out how to handle false-positives
#     f_name_write = './video_images/frame_{0:04d}'.format(int(n_frames - n_frames_counter)) + '.jpg'
#     cv2.imwrite(f_name_write, draw_img ) # debug
    # attach to video
    video.write(draw_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # display some processing information
    n_frames_counter -= 1
    t_img_end = time.time() - t_img_start
    print('Processing time for img: ', t_img_end)
    print('Remaining number of images to process: ', n_frames_counter)
    # calculate an estimate of the remaining processing time
    t_step = n_frames_counter * t_img_end / 60 # [min] remaining processing time
    t_remaining.append(t_step)
    t_remaining_array = np.array(t_remaining)
    t_remaining_average = np.average(t_remaining_array, 0)
    print('Estimated remaining processing time:', t_remaining, ' mins')
    
print('Video processing time: ', time.time() - t_video_start, 's') 
# When everything done, release the capture
video.release()
cap.release()
cv2.destroyAllWindows()

print('Script finished.')

## Writeup Vehicle Detection by Igor Gall

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/Window_boxes.png
[image4]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the lines TODO through TODO of the file called `VehicleDetectionFunctions.py`. There the function "get_hog_features()" extracts the histogram of gradients using the cv2-function "hog()".
In order to get a feeling with the different function arguments in hog(), I used the "src.py" file to read in images and non-vehicle images and apply the function with different parameters.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces `orientations`, `pixels_per_cell`, and `cells_per_block`.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and figured that orient = 9, pix_per_cell = 8, cell_per_block = 2, and the HSV color space are the parameters that work the best for me. Using orient = 9 I had enough directions to represent the shape of a vehicle. With pix_per_cell = 8 there are enough cells to see a shape similar to a car within the shape by human eye, which I believe is better to use for the machine. That might be a false deduction :-) The HSV color space provides the S channel that showed distinct features within the lessons for cars.
As observed below in Rubric 2 for sliding window search, the parameters turned out to work poorly for identifying robustly cars.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the SVC classifier from line TODO to line TODO. First of all, I used only the HOG features. Using the "extract_features()" function from 'VehicleDetectionFunctions.py' I accumulated the features for a subset of 1000 for each car and non-car images.
In the next step, I used the standard scaler to normalize the features in lines TODO-TODO. Using the normalized features, the subset of data was split into training and 10% validation samples. The actual training was performed in line TODO (svc.fit(X_train, y_train)). In order to figure out the achieved accuracy I used the score()-method in line TODO(svc.score(X_test, y_test)).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Using the "slide_window()" function from 'VehicleDetectionFunctions.py' that was used from the lesson, I received the windows of interest in line TODO (slide_window()). The function uses different window sizes and overlaps to create windows on the input image. I decided that I want to identify vehicle close, mid and far away from the ego vehicle. Using the software GIMP I found out that window sizes of (92, 92) for far-distanced, (144, 144) for mid-distanced, and (256, 256) for close-distanced vehicles is a reasonable choice.
For the overlap I thought that a dense map of windows is a good approach to detect vehicles, so I chose an overlap of 80%. It takes more calculation time, but my goal is to have a good identification performance which should be better with many windows.
Once I received the window coordinates, I used the search_windows()-function in line TODO (hot_windows = search_windows) to classify the windows as cars or non-cars. The underlying function is "single_img_features()". It uses the image features and assigns them to the trained classifier in the above rubric. The classifier performs a prediction on the image features in line TODO(clf.predict(test_features)) in file "VehicleDetectionFunctions.py".

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

On my machine, it took about four seconds to analyze the three window scales. So I decided to disregard the largest window size (256, 256)-pixels, in order to decrease the computational time.
As it turned out, my choice of parameters for the HOG parameters did not work well, because of many false positives. I had to change them and came up with the following new HOG parameters:
* color_space = 'YCrCb'
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL'
Here are some example images:

![alt text][image4]
As you see, there are still some false positives but that could be handled by the heat map approach.
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


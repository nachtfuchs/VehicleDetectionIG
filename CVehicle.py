''' @brief  A vehicle class that stores the detection properties and 
            provides methods.
'''
import numpy as np

class CVehicle():
    def __init__(self):
        self.detected = False # was the vehicle detected in the last iteration
        self.n_detections = 0 # Number of times this vehicle has been detected.
        self.n = 3 # hard coded. Amount of labels that will be taken for determining the best fit
        self.n_nondetections = 0 # number of consecutive times this vehicle has not been detected
        self.non_detection_limit = 5 # number of non-detections allowed until the car properties are reset
        self.xpixels = None # Pixel x values of last detection
        self.ypixels = None # Pixel y values of last detection
        self.recent_xfitted = [] # x position of the last n fits of the bounding box
        self.bestx = None # Average x position of the last n fits
        self.recent_yfitted = [] # y position of the last n fits of the bounding box
        self.besty = None # Average y position of the last n fits
        self.recent_wfitted = [] # width of the last n fits of the bounding box
        self.bestw = None # Average width of the last n fits
        self.recent_hfitted = [] #height of the last n fits of the bounding box
        self.besth = None # Average height of the last n fits
        # self.cog = (None, None) # Center of gravity of the best fit

    def new_detection(self, labels, i):
        ''' @brief This function deals with new detections.
        @input labels A return value of the "scipy.ndimage.measurements import label()"-function
        @input i      The label that was assigned to the new detection
        '''
        # set vehicle as detected
        self.detected = True
#         self.n_detections = 1 # the detections are increased in the add_box(method)
        # get the pixels where the label was found
        nonzero = (labels[0] == i).nonzero()
        self.xpixels = nonzero[1]
        self.ypixels = nonzero[0]
        # first detection goes into the buffer
        self.recent_xfitted.append(self.xpixels)
        self.recent_yfitted.append(self.ypixels)
        
        width = np.max(self.xpixels) - np.min(self.xpixels) # width is saved in the x pixels
        height = np.max(self.ypixels) - np.min(self.ypixels)# height is saved in the y pixels
        
        self.recent_wfitted.append(width)   # append, because it is a list of all past widths
        self.recent_hfitted.append(height)  # append, because it is a list of all past heights
        
    def add_box(self, labels, i):
        ''' @brief This function deals with new detections.
        @input labels A return value of the "scipy.ndimage.measurements import label()"-function
        @input i      The label that was assigned to the new detection
        '''
        # increase detection values
        self.detected = True
        self.n_detections += 1
        
        # reduce the non-detection counter if the vehicle was identified
        if self.n_nondetections > 0:
             self.n_nondetections -= 1
             
         # get the pixels where the label was found
        nonzero = (labels[0] == i).nonzero()
        self.xpixels = nonzero[1]
        self.ypixels = nonzero[0]
        
        # concatenate the newly detected pixels to the recent detections
        self.recent_xfitted.append(self.xpixels)
        self.recent_yfitted.append(self.ypixels)
        width = np.max(self.xpixels) - np.min(self.xpixels) # width is saved in the x pixels
        height = np.max(self.ypixels) - np.min(self.ypixels)# height is saved in the y pixels
        self.recent_wfitted.append(width)
        self.recent_hfitted.append(height)
    
    def calc_best(self):
        ''' @brief Calculate the best fit of the vehicle window over the last n labels
        '''
        # calculate the average height of the last n windows
        heights = self.recent_hfitted[-self.n:]
        self.besth = np.int(sum(heights)/ len(heights) ) # when height has less than self.n elements, it length has to be taken.
        widths = self.recent_wfitted[-self.n:]
        self.bestw = np.int(sum(widths) / len(widths) )
        
        # get the minimum values from the last n labels
        x_mins = []
        for el in self.recent_xfitted[-self.n:]:
            x_mins.append(np.min(el))
        
        y_mins = []
        for el in self.recent_yfitted[-self.n:]:
            y_mins.append(np.min(el))
        
        # calculate the best pixels
        self.bestx = np.int(sum(x_mins) / len(x_mins))
        self.besty = np.int(sum(y_mins) / len(y_mins))

    def not_detected(self):
        ''' @brief  This function deals with the vehicle properties, if it was not detected
                    in the current image. The idea is that the previous best fit is used
                    while the non-detection counter rises
        '''
        self.n_nondetections += 1
        
    def merge(self, car_to_merge):
        ''' @brief The goal is to merge the settings of two cars. self inherits the settings
        of the second argument
        @input car_to_merge A CVehcile object that will pass its features to self
        '''
        self.n_detections = 0
        self.n_nondetections = 0
        # assign the best values of the second object to the self object
        self.recent_xfitted.append(car_to_merge.bestx) 
        self.recent_yfitted.append(car_to_merge.besty)
        self.recent_hfitted.append(car_to_merge.besth)
        self.recent_wfitted.append(car_to_merge.bestw)
        # calculate the new best fit
        self.calc_best()


#     def calc_cog(self):
#         ''' @brief This function calculates coordinates of the center of gravity (cog) for the best fit.
#         '''
#         # todo: Deal with bestx = None
#         cog_x = self.bestx + self.bestw // 2
#         cog_y = self.besty + self.besth // 2
#         
#         self.cog = (cog_x, cog_y) #return value

# instantiation
# CVehicle = CVehicle()
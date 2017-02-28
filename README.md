# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)

[image1]: ./output_images/vehicle.png "Car Image"
[image2]: ./output_images/nonvehicle.png "Non-car Image"
[image3]: ./output_images/features.png "Features"
[image4]: ./output_images/scaled_features.png "Scaled Features"
[image5]: ./output_images/scaled_together.png "Scaled Features Seperatly"
[image6]: ./output_images/test6.jpg "Test Image"
[image7]: ./output_images/test6_found.png "Test Image Found"
[image8]: ./output_images/test6_heat.png "Heat Image"
[image9]: ./output_images/test6_final.png "Final Image"


#### 1. HOG feature detection and SVM classification


Initial step a vehicle detection is to have sufficient dataset of vehicle and non-vehicle images from various conditions. For datasets following databases are used during this project:

Labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. 

| Vehicle Image            | Non-vehicle Image      | 
|:------------------------:|:----------------------:| 
|![Car Image][image1]      |![Non-car Image][image2]| 

[HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) is a feature extraction method that counts occurances of histogram orientation for a given portion of an image. It firstly introduced by [Dalal et. al.](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) at their study of human detection and it is really helpful computer vision technique for shape based histogram feature engineering. Detailed explanation of HOG for code implementation can be found at [scikit library](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog).

There is other feature extraction methods as well. Rescaling images to 32x32 and making bin spatial could be another method. And also objects could be characterized by their color. In below color, bin spatial and hog feature plots can be seen.

![Features][image3] 

As one can see all feature plotings in different scaling domain and they should scaled for further feature classification. In below apporach can be found; scaled together and scaled seperatly and then combined:

![Scaled Features][image4] 
![Scaled Features Seperatly][image5] 

For a further step this scaled and combined features matrices should be used for classifications. Using support vector machines and label matrices of car and non-car images it is possible. By using `LinearSVC()` class and `svc.fit` function with only using HOG features %99 test accuracy is achieved with %20 splitted test set.


#### 2. Finding cars


`find_cars` function is where trained classifier used for our purpose. This function calculates HOG features for all channels of a picture. Here pictures are every frame of a video and they cropped to only see certain scene (there is no vehicle at top or bottom). To speed up calculations it's calculates all HOG features at once and takes results window by window as subimage and test whether there is a car or not. If classifier results a 1 then searched subimage boundaries recorded in `box_list`. Due searching for different sized boxes, number of car found boxes could be more than one for same car. This fuctionality gives us to eliminate false positives and also gives us an understanding about how certain we are at the classification. 


| Test Image               | Bounding Boxes                 |
|:------------------------:|:------------------------------:|
|![Test Image][image6]     |![Test Image Found][image7]     |

#### 3. Heat map


There is three functions in pipeline for heatmap plotting. Heatmap help us to understand false positives and classification certainty. `add_heat` function adds 1 as a value for every pixel at detected boxes. `apply_threshold` function removes values lower than threshold for every pixel, this help us to understand false positives. And finally `draw_labeled_bboxes` draws rectangles according to found labels from `scipy.ndimage.measurements` class's `label` function.


| Test Image               | Heat Image                     |
|:------------------------:|:------------------------------:|
|![Test Image][image6]     |![Heat Image][image8]           |


#### 4. Results of test images


Here's a example test image result, found car bounding boxes and heat map result

| Test Image               | Final Result                   |
|:------------------------:|:------------------------------:|
|![Test Image][image6]     |![Test Image Final][image9]     |



#### 5. Results of project video


Here's a [link to my video result](./project_video_output_v1.mp4)


#### 6. Discussion


In this project; color, bin spatial and HOG feature creations are investigated. SVM classifier is implemented and it is trained with car and non-car images. For vehicle detection and tracking alogrithm development, trained classifier is used with window search algorithm. Feature matrix is calculated at once and it is subsampled images asked for the result from trained classifier. According to returned result of a classifier, car found boxes rewarded by 1 and heat map is generated according to `label` matrix and false positives are eliminated accordingly.
 

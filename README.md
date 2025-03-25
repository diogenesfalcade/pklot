# This is a project based on the the "PKLot – A robust dataset for parking lot classification" research, available on https://doi.org/10.1016/j.eswa.2015.02.009
### I used the database available in the article to create a classifier using LBP and KNN with C++.
### The project includes 4 main steps
### 1 - Crop images (cropImages.cpp)
### 2 - Generate histogram with LBP (LBPHistogram.cpp)
### 3 - Split between training and test set (split_t_t_norm.cpp)
### 4 - Apply KNN (knn.cpp)

### For step 1, I used the OpenCV library to crop the images accordigly to what the Autors described with the XML file provided. Also, I used libxml to process the data from the XML file.
### On step 2, as I needed to process images, again I used the OpenCV lib to first calculate the LBP of the image using grey scale and then compute the LBP histogram to generate the features with 256 values
### After storing the features on a CSV file, for the step 3 I splited data between train and test considering a 50/50 split, also, I have applied a normalization using the mix max method. This took into account that images from a particular day could not be shared between train and test sets.
### For the last, I calculated the KNN for all the test values, reaching a 92.30% accuracy

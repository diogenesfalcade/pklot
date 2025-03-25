# Project based on "PKLot – A robust dataset for parking lot classification" research
### Available at: https://doi.org/10.1016/j.eswa.2015.02.009

## Goal:
### Create a classifier to identify whether a parking spot is empty or occupied.
### Used the dataset from the article and implemented it using LBP (Local Binary Patterns) and KNN (K-Nearest Neighbors) in C++.

## Project Steps:
### 1 - Crop images (cropImages.cpp)
### 2 - Generate LBP histogram (LBPHistogram.cpp)
### 3 - Split data into training and test sets (split_t_t_norm.cpp)
### 4 - Apply KNN classification (knn.cpp)

### Implementation Details:
### Step 1:
### - Used OpenCV to crop images based on the coordinates provided in the XML file.
### - Processed XML data using libxml.

### Step 2:
### - Used OpenCV to convert images to grayscale and compute LBP.
### - Generated a 256-bin LBP histogram to extract features.
### - Stored features in a CSV file.

###Step 3:
### - Split data 50/50 between training and test sets.
### - Applied min-max normalization.
### - Ensured images from the same day were not shared between sets to avoid bias.

### Step 4:
### - Implemented KNN classification.
### - Achieved 92.30% accuracy on the test set.

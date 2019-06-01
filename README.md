# Chest-X-Ray-Classification-CNN

## Code Requirements

* Numpy
* Pandas
* cv2
* Seaborn,matplotlib
* Keras
* Pathlib
* Sklearn

## Description

This is an image classification problem on Kaggle Datasets.The dataset contains 3 folders -train, validation and test

![alt text](https://github.com/pshar33/Chest-X-Ray-Classification-CNN-/blob/master/Plots.png)


## Breakdown of the code:

1. Loading the dataset: Load the data and import the libraries.
2. Data Preprocessing:
     * Reading the images stored in 3 folders(Train,Val,Test).
     * Plotting the NORMAL and PNEUMONIA images with their respective labels.
3. Data Augmentation: Augment the train,validation and test data using ImageDataGenerator
4. Creating and Training the Model: Create a cnn model in KERAS.
5. Evaluation: Display the plots from the training history.
6. Prediction: Run predictions with model.predict
7. Conclusion: Comparing original labels with predicted labels and calculating confusion matrix

## Accuracy and loss plots





## Results:

- The accuracy for the test dataset came out to be  . 
- The link to my kaggle kernel is   which is a jupyter notebook.Please feel free to leave a feedback or comment.

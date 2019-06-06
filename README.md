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

This is the comparative plots of Normal and Pneumonia chest X Rays. 


## Breakdown of the code:

1. Loading the dataset: Load the data and import the libraries.
2. Data Preprocessing:
     * Reading the images stored in 3 folders(Train,Val,Test).
     * Plotting the NORMAL and PNEUMONIA images with their respective labels.
3. Data Augmentation: Augment the train,validation and test data using ImageDataGenerator
4. Creating and Training the Model: Create a cnn model in KERAS.
5. Evaluation: Display the plots from the training history.
6. Prediction: Run predictions with model.predict
7. Conclusion: Comparing original labels with predicted labels and calculating recall score.

## Accuracy and loss plots


![alt text](https://github.com/pshar33/Chest-X-Ray-Classification-CNN-/blob/master/loss%2Caccuracy%20plots.png)
![alt text](https://github.com/pshar33/Chest-X-Ray-Classification-CNN-/blob/master/loss%2Caccuracy%20plots2.png)


## Results:

- The accuracy for the test dataset came out to be > 85% .
- The accuracy for the validation dataset came out to be > 93 % .
- The link to my kaggle kernel is https://www.kaggle.com/parthsharma5795/chest-x-ray-classification-cnn-keras  which is a jupyter notebook.Please feel free to leave a feedback or comment and upvote on KAGGLE if you found it to be helpful !!

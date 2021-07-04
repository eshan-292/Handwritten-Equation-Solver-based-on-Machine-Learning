# Handwritten-Equation-Solver-based-on-Machine-Learning

The task is divided into two parts. For the first part, given a labelled dataset about whether a given equation is infix/prefix/postfix, it predicts the type of equation for new test data. For the second part, our model identifies the each of the three characters in a equation and then performs the required operation on them giving the corresponding output. 

METHODS:
	As far as the model part is concerned, we have used CNN to train our models in both the parts. We have added multiple convolutional layers, accompanied with pooling and flattening them to achieve better results.

	In the first part, we are provided with labelled data, so we directly ran our CNN model on it. For the second part, we first manually segregated around 600 images into their valid classifications(10 Digits and 4 Symbols) and trained our CNN model over them. Then we used our trained model to predict the classifications for around 3000 images(by recognising the characters) in the given dataset and sorted them accordingly(automated). Now we trained our model on the large segregated dataset obtained to achieve a stronger model and therefore better accuracy.

	Then for every test image we first split the image into three parts and let our model predict the character in each part. Based on the type of operation(infix/prefix/postfix) we evaluate the expression and store the final results into a csv file.

HOW TO USE:
->The annotations.csv file contains labelled data for the first part.
->model_final_1 and model_final_2 contain the trained models(along with the saved weights) for first and second parts respectively.
->NEW.zip contains the segregated data used for training our digit recognition model.
.>Requirements.txt contains all the dependencies needed for the project.
->train1 and train2 files contain the code used for training the models.
->inference1 and inference2 include the codes used for testing the models.
->Final.ipynb is the Jupyter Notebook consisting of all the code together along with the results of each step.

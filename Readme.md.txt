METHODS:
	As far as the model part is concerned, we have used CNN to train our models in both the parts. We have added multiple convolutional layers, accompanied with pooling and flattening them to achieve better results.

	In the first part, we are provided with labelled data, so we directly ran our CNN model on it. For the second part, we first manually segregated around 600 images into their valid classifications(10 Digits and 4 Symbols) and trained our CNN model over them. Then we used our trained model to predict the classifications for around 3000 images(by recognising the characters) in the given dataset and sorted them accordingly(automated). Now we trained our model on the large segregated dataset obtained to achieve a stronger model and therefore better accuracy.

	Then for every test image we first split the image into three parts and let our model predict the character in each part. Based on the type of operation(infix/prefix/postfix) we evaluate the expression and store the final results into a csv file.

->We have made two train files train1.py and train2.py for parts 1 and 2 respectively.
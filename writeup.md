#Traffic Sign Recognition

#Writeup

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

##Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my project code

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I used the numpy capabilities to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![bar_chart](bar_chart.png)

X axis represents the types of traffic signs. Y axis represents the number of occurrences of a particular type.

Below is a random traffic sign from the dataset.

![random_sign](random_sign.png)

According to the labels provided with the training data, the sign is of type 17 'No entry'.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I tried several ways to preprocess images including converting to gray scale, converting to Luv color space, and enhancing image contrast. However, none of these method gave improvement in image classification.

Finally, I decided to use gradient of the image because gradient shows how neighbor pixels are different. This should be really useful because traffic signs have sharp borders and distinct shapes within. All the following training and recognition is done based on image gradients.

Here is an example of a traffic sign image before and after preprocessing.

![before_preproc](before_preproc.png)

![after_preproc](after_preproc.png)

Sign 10 'No passing for vehicles over 3.5 metric tons'.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook.

My final model consisted of the following layers:

| Layer         | Description   |
|:-------------:|:-------------:|
| Input      | 32x32x3 RGB image |
| Convolution 5x5      | 1x1 stride, valid padding, outputs 28x28x6      |
| tanh |       |
| Average Pooling | Kernel 2x2, stride 2x2, outputs 14x14x6      |

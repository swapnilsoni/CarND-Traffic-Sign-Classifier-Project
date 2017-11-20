# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/traffics_signals.png "Traffic Signals"
[image2]: ./images/bars_training_size.png "Training Size"
[image3]: ./images/german_traffic_images.png "German traffic images"
[image4]: ./images/German_traffics_predictions.png "Prdicted German traffics signal images"
---

#### 1. Source Code
[project code](https://github.com/swapnilsoni/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Data set [Download](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)
* The size of training set is 31319
* The size of the validation set is 3480
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Traffic signals][image1]
![Number of instances of each class][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

Pre-processing of imgaes is part of the model archetecure itself:

1) As a first step, I decided to convert the images data type from uint8 to float32
  tf.cast(images, tf.float32)

2) Convert images to gray scale. Can not show it because I am using tensorflow api.
  tf.image.rgb_to_grayscale(images)

3) As a last step, I normalized the image data because using 
   tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        								| 
|:---------------------:|:---------------------------------------------------------:| 
| Input         		| 32x32x1 Gray scale image   								| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 32x32x6				|
| RELU					|															|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6  							|
| Convolution 5x5	    | 16x16x6 input,1x1 stride,valid padding outputs 16x16x16	|
| RELU					|															|
| Max pooling	      	| 2x2 stride,  outputs 8x8x16  								|
| Flatten		      	| Convert into 1 dimension  								|
| Fully connected		| 294 input, and 100 output									|
| Fully connected		| 100 input, and 43 output									|
|						|															|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* Convert output label to one_hot code
* Initial: Learning rate = 0.001
* Optimizer: AdamOptimizer. This function will automatically tune the learning rate
* Loss function softmax_cross_entropy_with_logits
* Batch Size 1000
* EPOCH size: 100


### Test a Model on New Images

#### 1. Choose 10 German traffic signs found on the web and provide them in the report. 

Here are five German traffic signs that I found on the web:

![10 images][image3]

#### 2. Discuss the model's predictions on these new traffic signs. 

Here are the results of the prediction:
![10 images][image4]
Real-Time Age, Gender & Ethnicity Prediction Model

       -By Aayush Bhat

-Introduction

**PS: I edited the jupyter notebook a bit to make the project easier to implement as i realized it one had to run all the cells to get the project to run, now we need to run only four cells. So pls checkout the jupyter notebook in this github repo instead of the one provided in the google form.**

Facial recognition and analysis is a highly useful and evolving sub-group of Computer Vision and Machine Learning in general. Use of CNN has resulted in extremely accurate prediction/classification models. This has various applications from unlocking your smartphone to high-tech security systems.

My project is a starting step in this field of ML and I have tried to use as many things I have learnt from CSOC as possible to create a model which takes feed from your webcam and predicts your age/gender and ethnicity to decent accuracy levels. Upon rudimentary testing, I found that the model works well in most cases especially if the person is sitting in a well-lit room.


-Data Preprocessing & Exploratory Analysis


I used the UTKFace dataset for training my models which contained 23708 images with age, gender and ethnicity labels in the filenames. I used the split functions to gather the data and stored them in their respective numpy arrays. 

I resized and normalized the images for obvious reasons.

I encoded the age data into ranges:

1) [0-8]
2) [9-18]
3) [19-29]
4) [29-60]
5) [60+]
I had to choose the data ranges in such a way that specific ranges aren't very skewed but that posed a challenge since most of the entries are in the 20-35 age range. But I tried my best to keep the data balanced.

The gender values 0 represent Male while 1 represent Female.

The ethnicity values represent:
0 - White
1 - Black
2 - Asian
3 - Indian
4 - Others

I have provided the distribution of the dataset in the notebook.


Creating the Models


I created three separate models for age, gender and ethnicity and used a varying number of convolution layers and activation functions which best suited the corresponding datasets.
I wanted to train the models for longer times but activating my GPUs in the tensorflow environment was a massive pain. So I had to be content with training only for 10 epochs but I feel the models could have performed better if they trained for longer times.


Real-time Video Capture


In this part of the project, I first used opencv for video capturing and used EVERY frame of the captured video as an image. Using OpenCV's face recognition model HaarCascades, the model extracts the face from every frame and
 this image is then feeded to all of my models and the predictions are put to the frame using the putText() command.

This was fairly simple and with that my project is concluded.

Now, I want to look into more techniques such as transfer learning, mix-up my dataset by rotating it to increase variety and thus have a bigger training pool. I also want to test out more configurations of CNN layers to maximise accuracy.
     




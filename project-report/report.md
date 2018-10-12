# Image Classification using k-means on TensorFlow

Selahattin Akkas
Indiana University
Bloomington, Indiana
sakkas@iu.edu

##Abstract

##Keywords
image classification, k-means, YFCC100M, TensorFlow

##1. Introduction
The project goal is clustering the Yahoo Flicker Creative Commons 100 Million (YFCC 100 Million) using k-means on TensorFlow. Since data set is very large and some media has no tags, it will be hard to measure the accuracy. Yahoo also shares a subset of the dataset which has 10 tagged classes. In this work 10 class dataset will be used.

##Implementation
####Technologies Used
#####TensorFlow
TensorFlow is a  Machine Learning/Deep Learning framework developed by Google. It is continuation of DisBlief which is Google’s internal use framework.

TensorFlow is widely used for Machine Learning/Deep Learning applications. It is easy to develop deep learning applications on TF. After training, applications can be easily deployed and used even on mobile phones [@fa18-523-68-url-tensorflow-lite].

###Dataset
Yahoo Flickr Creative Commons 100 Million (YFCC100m) dataset consists ~100 million photos(99.2 million) and videos(0.8 million). Medias in the dataset carry Creative Commons license [@fa18-523-68-Kalkowski2015]. Some medias have tags but in general the data is unlabeled. Therefore, total number of class is unknown and it will make the clustering harder. There is a subset of the dataset which have 10 classes. To see accuracy performance, this subset will be used in the project. 

### Milestones and Time Plan

**10/12 - 10/18:**

1. Clear the data if needed
2. Extract the using VGG19  (It will be done one time and the extracted features will used many times) 

**10/19 - 10/25:**
3. Tensorflow installation
4. Run the  built-in k-means on single node and decide the data size. (I need to get the results in reasonable time)
5. Run the built-in k-means on 2-3 nodes.

**10/26 - 11/08:**
6. Implement k-means to TensorFlow and run it on single node
7. Run own implementation on multiple nodes. 

**11/09 - 11/26:**
8. Fix the problems
9. Write the final paper


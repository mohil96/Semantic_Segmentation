# Semantic Segmentation for Self-Driving Cars
### Introduction
In this project we label the pixels of a road in images using a SegNet model.
There are 13 classes that the model outputs, but only the road class is used for lane detection.

=======

##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

You may also need [Python Image Library (PIL)](https://pillow.readthedocs.io/) for SciPy's `imresize` function.

##### Dataset
Using Cityscapes Dataset

##### SegNet

Direct adoption of classification networks for pixel-wise segmentation yields poor results mainly because
max-pooling and sub-sampling reduce feature map resolution and hence output resolution is reduced.
SegNet is an image segmentation architecture that uses an encoder-decoder type of architecture. This is
a Fully Convolutional Network.

The encoder is usually a pre-trained classification network like VGG or ResNet followed by a decoder
network. In SegNet, it uses a pre-trained VGG16 model for its encoder part. All the 13 convolution layers
are of VGG16 (16 in VGG16 refers to the number of layers that have weights) are used.

In the decoding process, to up-sample the layers, SegNet uses the “max pooling indices” at the
corresponding encoder layer are recalled. This makes the training process easier since the network need
not learn the up-sampling weights again.
For every encoder layer, there exists a corresponding decoder network to up-sample the image to its
original size. A pixelwise classification layer i.e., SoftMax unit is followed by decoder network.

##### Inference Optimization

Neural networks are typically trained with a single precision 32-bit floating point number. However, we
do not need single precision for all the tasks, especially for something like inferencing. Lowering
precision can affect the accuracy while training the network during back propagation but does not have
much effect post-training.

Quantization involves conversion of floating point values to integers by discretizing over a range by linear
conversion. This reduces precision to increase performance during inferencing.
There are multiple post-training techniques available that are mentioned in detail [here](https://www.tensorflow.org/model_optimization/guide/quantization/post_training).

##### Results

Mean Intersection over Union (mIoU) : 60.10
Training accuracy : 86.66 %
Validation accuracy : 81.73%




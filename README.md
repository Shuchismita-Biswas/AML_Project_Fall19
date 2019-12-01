This repository was made to meet the final project requirements of course Advanced Machine Learning (CS5824), Fall 2019.

## Team members
Shuchismita Biswas, Sarthak Gupta, Sanij Gyawali, Sagar Karki

# Classification of Time-Series Images Using Deep Convolutional Neural Networks
A time-series refers to a sequence of data points, ordered temporally. Time-series analysis finds many real-world applications, in fields like weather forecasting, stock markets, biomedical signal monitoring, video processing and industrial instrumentation. This has motivated efforts and research into time-series classification (TSC) tasks that assign a label <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_n}"/> to a time-series <img src="https://latex.codecogs.com/svg.latex?\Large&space;x_n}"/>. Traditionally TSC methods consist of spectral analysis and wavelet analysis (frequency-domain), and, auto-correlation, auto-regression and cross-correlation (time-domain). With the increased computation power and availability of large data sets, Deep Neural Networks (DNNs) have been employed for this purpose more recently. Recurrent Neural Networks is one such specialized DNNs that was developed for TSC and has been used extensively.

Convolutional Neural Networks (CNNs) are also a specialized kind of DNNs that have been traditionally employed for image related machine learning applications. If there were ways to convert time-series to images CNNs could be applied. This project explored the above possibility by studying and comparing two methods of converting time-series into images - Recurrence Plots (RPs) and Gramian Angular Summation Fields (GASF) for the purpose of TSC using CNNs.

## Convolutional Neural Networks
CNNs like regular DNNs consist of a input layer, hidden layers and an output layer. Training a CNN also has a similar purpose as a regular DNN - to minimize the loss function measured at the output layer. The main difference between the CNNs and DNNs lies in how this information flows through a CNN.

Unlike a regular DNN which use matrix dot products, the information between layers in a CNN can be a result of the convolution operation. This is specifically useful for images for which a convolution can be seen as a rolling matrix dot products over smaller portions of the image. Additionally, multiple such moving matrices can be stacked together to add a dimension of depth in addition to width and height.

A regular DNN [3] :
<img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

A CNN [3] :
<img src="http://cs231n.github.io/assets/cnn/cnn.jpeg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

Additionally, CNNs include a Pool layer for downsampling the information and a fully-connected output layer to produce output values corresponding to each of the  classification classes.
## Recurrence Plots
## Gramian Angular Summation/Difference Fields
## Experiments and Results
## Summary

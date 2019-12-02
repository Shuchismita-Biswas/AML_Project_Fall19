This repository was made to meet the final project requirements of course Advanced Machine Learning (CS5824), Fall 2019.

## Team members
Shuchismita Biswas, Sarthak Gupta, Sanij Gyawali, Sagar Karki

# Classification of Time-Series Images Using Deep Convolutional Neural Networks
A time-series refers to a sequence of data points, ordered temporally. Time-series analysis finds many real-world applications, in fields like weather forecasting, stock markets, biomedical signal monitoring, video processing and industrial instrumentation. This has motivated efforts and research into **time-series classification (TSC)** tasks that assign a label <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_n}"/> to a time-series <img src="https://latex.codecogs.com/svg.latex?\Large&space;x_n}"/>. Traditional TSC methods may be categorized into two broad classes- frequency-domain (spectral analysis, wavelet analysis etc) and time-domain (auto-correlation, auto-regression, cross-correlation etc) methods. More recently deep neural networks (DNN) have been applied to TSC tasks successfully. 

Recent advances in the field of computer vision have developed efficent DNNs, like **Convolutional Neural Networks (CNN)** for image classification. In recent literature, some papers have proposed image embedding of time series data so as to leverage image classification algorithms for the TSC task. In this project, we reproduce two image embedding methods for time series data- **Recurrent Plots (RP)** [1] and **Gramian Angular Summation Field (GASF)** [2] and use a CNN to classify the generated images. We test the classification algorithm using two datasets from the UCR dataset archive [3] and verify the results reported in [1].

## Encoding Time-Series into Images
The idea of imaging time-series entails training machines to *visually* recognize, classify and learn temporal structures and patterns. Here, we describe two methods that this project has explored. 

## Recurrent Plots (RP)
Time series data are characterized by distinct behavior like periodicity, trends and cyclicities. Dynamic nonlinear systems exhibit recurrence of states which may be visualized through RPs. First introduced in [4], RPs explore the <img src="https://latex.codecogs.com/svg.latex?\Large&space;m}"/>-dimensional phase space trajectory of a system by representing its recurrences in two dimensions. They capture how frequently a system returns to or deviates from its past states. Mathematically, this may be expressed as below.

$$x_n$$


<img src="https://latex.codecogs.com/svg.latex?\Large&space; R_{i,j}=||\Vec{s_i}-\Vec{s_j}||, \quad i,j=1,2,\dots K}"/>

Here, $\Vec{s_i}$ and $\Vec{s_j}$ represent the system states at time instants $i$ and $j$ respectively. $K$ is the number of system states considered. In the original RP method, the $R$ matrix is binary, i.e. its entries are $1$ if the value of $||\Vec{s_i}-\Vec{s_j}||$ is above a pre-determined threshold and $0$ otherwise. We do away with the thresholding since unthresholded RPs capture more information. Images so obtained capture patterns which may not be immediately discernible to the naked eye. A detailed procedure for constructing a RP plot of a simple time series is shown in Fig.~\ref{fig:RP}.

## Convolutional Neural Networks
CNNs like regular DNNs consist of a input layer, hidden layers and an output layer. Training a CNN also has a similar purpose as a regular DNN - to minimize the loss function measured at the output layer. The main difference between the CNNs and DNNs lies in how this information flows through a CNN.

Unlike a regular DNN which use matrix dot products, the information between layers in a CNN can be a result of the convolution operation. This is specifically useful for images for which a convolution can be seen as a rolling matrix dot products over smaller portions of the image. Additionally, multiple such moving matrices can be stacked together to add a dimension of depth in addition to width and height.

A regular DNN [3] :
<img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg" width="50%"
     alt="DNN"
     title="Example of a DNN" />

A CNN [3] :
<img src="http://cs231n.github.io/assets/cnn/cnn.jpeg"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />

Additionally, CNNs include a Pool layer for downsampling the information and a fully-connected output layer to produce output values corresponding to each of the  classification classes.
## Recurrence Plots
## Gramian Angular Summation/Difference Fields
## Experiments and Results
## Summary

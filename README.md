This repository and project report was made to meet the final project requirements of course Advanced Machine Learning (CS5824), Fall 2019

## Team members
Shuchismita Biswas, Sarthak Gupta, Sanij Gyawali, Sagar Karki

# Classification of Time-Series Images Using Deep Convolutional Neural Networks
The term time-series is used to refer to a sequence of data with temporal correlation. Such series are omnipresent in the world - the stock prices, videos, music, etc. This has motivated efforts and research into time-series classification (TSC) tasks that assign a label <img src="https://latex.codecogs.com/svg.latex?\Large&space;y_n}"/> to a time-series <img src="https://latex.codecogs.com/svg.latex?\Large&space;x_n}"/>

Traditionally TSC methods consist of spectral analysis and wavelet analysis (frequency-domain), and, auto-correlation, auto-regression and cross-correlation (time-domain). With the increased computation power and availability of large data sets, Deep Neural Networks (DNNs) have been employed for this purpose more recently. Recurrent Neural Networks is one such specialized DNNs that was developed for TSC and has been used extensively.

Convolutional Neural Networks (CNNs) are also a specialized kind of DNNs that have been traditionally employed for image related machine learning applications. If there were ways to convert time-series to images CNNs could be applied. This project explored the above possibility by studying and comparing two methods of converting time-series into images - Recurrence Plots (RPs) and Gramian Angular Summation Fields (GASF) for the purpose of TSC using CNNs.

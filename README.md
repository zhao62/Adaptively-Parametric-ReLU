# Adaptively-Parametric-ReLU
Although the adaptively parametric ReLU (APReLU) was originally applied to vibration-based fault diagnosis, it can be applied in other applications, such as image classification, as well. In this code, the APReLU is implemented using TensorFlow 1.0.1 and TFLearn 0.3.2, and applied for image classification.

![The basic idea of APReLU](https://github.com/zhao62/Adaptively-Parametric-ReLU/blob/master/Basic-idea-of-APReLU.jpg)

Abstract:
Vibration signals under the same health state often have large differences due to changes in operating conditions. Likewise, the differences among vibration signals under different health states can be small under some operating conditions. Traditional deep learning methods apply fixed nonlinear transformations to all the input signals, which has a negative impact on the discriminative feature learning ability, i.e., projecting the intra-class signals into the same region and the inter-class signals into distant regions. Aiming at this issue, this paper develops a new activation function, i.e., adaptively parametric rectifier linear units, and inserts the activation function into deep residual networks to improve the feature learning ability, so that each input signal is trained to have its own set of nonlinear transformations. To be specific, a sub-network is inserted as an embedded module to learn slopes to be used in the nonlinear transformation. The slopes are dependent on the input signal, and thereby the developed method has more flexible nonlinear transformations than the traditional deep learning methods. Finally, the improved performance of the developed method in learning discriminative features has been validated through fault diagnosis applications.

Reference:
Minghang Zhao, Shisheng Zhong, Xuyun Fu, Baoping Tang, Shaojiang Dong, Michael Pecht, Deep Residual Networks with Adaptively Parametric Rectifier Linear Units for Fault Diagnosis, IEEE Transactions on Industrial Electronics, 2020,  DOI: 10.1109/TIE.2020.2972458 

https://ieeexplore.ieee.org/document/8998530

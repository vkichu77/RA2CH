# RA2CHALLENGE
<h2>Deep Learning Ensembles for The Measurement Of Joint Scores Of Rheumatoid Arthritis </h2>

Krishnakumar Vaithinathan<sup>1</sup>, Julian Benadit Pernabas<sup>2</sup>

<sup>1</sup>Department of Computer Engineering, Karaikal Polytechnic College, Varichikudy, Karaikal-609609, India.

<sup>2</sup>Department of Computer Science and Engineering, Faculty of Engineering, CHRIST(Deemed To be University), Kengeri Campus, Kanmanike,Bangalore,560074, Karnataka,India.

Introduction:
<h4>
The proposed method uses a multi-tier of deep learning models for various processes like background removal, dimensionality reduction, bone approximation with thresholding, deep regressors to predict various Rheumatoid Arthritis(RA) scores. This method is created for participating in international competition RA2-DREAM Challenge conducted at University of Alabama at Birmingham (https://www.synapse.org/#!Synapse:syn20545111/wiki/594083) between November 4, 2019 to May 14, 2020. My ranking in three sub challenge are 11, 13 and 11. Team name: vkichuDL. The results can be checked at https://www.synapse.org/#!Synapse:syn20545111/wiki/597246. Implemented in Python/Tensorflow. 
</h4>
The figure.1 below illustrates the implementation of the proposed model. 

![Alt text](https://github.com/vkichu77/RA2CH/blob/master/imgs/blockdiag.jpg?raw=true "BlockDiagram")

Various models are developed from different operations in the preprocessing and the regression phases respectively.

Pre-processing Phase:
Here, the information in scans are enhanced so that the data pattern are easily captured by the regressors. The steps below are performed for LF, RF, LH and RH region groups respectively.
1.	Resize scans to dimension 128 x 128.
2.	Create multiple autoencoders[1] with five different l2 regularization parameters (1e-09, 1e-08, 1e-07, 1e-06, 1e-05).
3.	Calculate mean image from the outputs of above autoencoders.
4.	Perform adaptive thresholding and Contrast Limited AHE (CLAHE) for all images to remove background.
5.	These images are used as input for a dimensionality reduction autoencoder to resize the images as 32x32.

Regression Phase:
1.	DNN regression technique is used.
2.	Optimized on four independent databases created by rotation and shifting from the images obtained from the dimensionality reduction autoencoder.
3.	Stochastic gradient algorithm is used.
4.	The learning rates that are sequentially used on the databases are 0.1, 1e-3, 1e-7, 1e-11, 1e-15.
5.	The trained model with learning rate 1e-15 is applied on the test data.

Discussion:
	The preprocessing step is used on all the 367 training images. The regression phase is applied on individual regional groups like LH, LF, RH, RF. The following table shows the weighted RMSE of joint scores of RA by our method.
submission ID 9705546
SC1: 1.4869
SC2: 0.7536
SC3: 0.6523
To conclude, the results shows that the accuracy of the deep learning models is very moderate when the whole scan images are used.

Please cite this article for this code:

<b>
Sun D, Nguyen TM, Allaway RJ, Wang J, Chung V, Thomas VY, Mason M, Dimitrovsky I, Ericson L, Li H, Guan Y. A crowdsourcing approach to develop machine learning models to quantify radiographic joint damage in rheumatoid arthritis. JAMA network open. 2022 Aug 1;5(8):e2227423-.
</b>

<br/><br/><br/><br/>
References:
1.	Yasi Wang, Hongxun Yao, Sicheng Zhao, Auto-encoder based dimensionality reduction, Neurocomputing, 2016

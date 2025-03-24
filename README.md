# HCNS:A Deep Learning Model for Identifying Essential Proteins Based on Hypergraph Convolution and Sequence Features
The HCNS model, which integrates the Hypergraph Convolutional Network (HGCN) module, the Seq-CNN-MB-NAG feature extraction module, and the Multi-Layer Perceptron (MLP) recognition module, significantly enhancing the accuracy of essential protein identification.Experimental results show that the HCNS model outperforms existing methods, achieving an accuracy of ***93.38\%***, with an Area Under the Curve (AUC) of ***98.33\%*** and an Area Under the Precision-Recall Curve (AUPR) of ***97.16\%***, demonstrating its potential in essential protein identification. 
## 🛠️（1）Compilation environment
* **python：** 3.8；
* **pytorch：** 1.12；
* **pycharm：** 2020.3；
* For details of some other third-party libraries, see the files in the code folder；
## 🗓️（2）File Description
* **database：** Stores the data files used by the model；
* **model：** Store each sub-model code；
* **utils：** Store some data processing and some codes to implement other functions；
## 🎯（3）Run
After importing the project, configure the environment required by the project, and run the model_test.py file directly to get accurate prediction results for the test set.

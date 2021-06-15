___
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues) [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FUW-CIA&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=users&edge_flat=false)](https://hits.seeyoufarm.com)
[![License](https://img.shields.io/pypi/l/mia.svg)]() 
<a href="https://https://github.com/UW-CIA/Extensive_Evaluations/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/UW-CIA/Extensive_Evaluations"></a>
<a href="https://github.com/kaiiyer/UW-CIA/Extensive_Evaluations"><img alt="GitHub forks" src="https://img.shields.io/github/forks/UW-CIA/Extensive_Evaluations"></a>
<a href="https://github.com/UW-CIA/Extensive_Evaluations/graphs/contributors" alt="Contributors">
<img src="https://img.shields.io/github/contributors/UW-CIA/Extensive_Evaluations" /></a>
<a href="https://github.com/UW-CIA/Extensive_Evaluations/graphs/stars" alt="Stars">
<img src="https://img.shields.io/github/stars/UW-CIA/Extensive_Evaluations" /></a>
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields)](http://makeapullrequest.com)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v3/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)


# Extensive_Evaluations
*Running 10+ models with different arcfhitecture and parameters in anomaly detection use cases on the HMOG dataset with deep feature extratctors and different realistic experiment settings* 
Models include: 
* VAE (Variational Auto Encoder) 
* b-VAE (Beta Variational Auto Encoder) 
* KNN (K-Nearest neighbors)
* ABOD (Angle-based Outlier Detection)
* OCSVM (One-class Support Vector Machine) 
* HBOS (Histogram-based Outlier Score)
* PCA  (Principal Component Analysis) 
* MCD (Minimum Covariance Determinant)
* CBLOF (Cluster-Based Local Outlier Factor)
* LOF (Local Outlier Factor)
* IForest (Isolation Forest)
* FeatureBag (Feature Bagging) 
* SCNN (Siamese Convolutional Deep Neural Network) as a deep feature extractor 

<br> <br> 
Part of the following effort: 
**HARNESSING THE POWER OF GENERATIVE MODELS FOR MOBILE CONTINUOUS AND IMPLICIT AUTHENTICATION**

Interconnecting the following works: 
* Generative AI Models
* Continous Authentication 
* Implicit Authentication 
* Outlier Detection 

### Abstract 
Authenticating a user’s identity lies at the heart of securing any information system.
A trade off exists currently between user experience and the level of security the system
abides by. Using Continuous and Implicit Authentication a user’s identity can be verified
without any active participation, hence increasing the level of security, given the continuous
verification aspect, as well as the user experience, given its implicit nature.
This thesis studies using mobile devices inertial sensors data to identify unique movements and patterns that identify the owner of the device at all times. We implement,
and evaluate approaches proposed in related works as well as novel approaches based on a
variety of machine learning models, specifically a new kind of Auto Encoder (AE) named
Variational Auto Encoder (VAE), relating to the generative models family. We evaluate
numerous machine learning models for the anomaly detection or outlier detection case of
spotting a malicious user, or an unauthorised entity currently using the smartphone system. We evaluate the results under conditions similar to other works as well as under
conditions typically observed in real-world applications. We find that the shallow VAE
is the best performer semi-supervised anomaly detector in our evaluations and hence the
most suitable for the design proposed.
The thesis concludes with recommendations for the enhancement of the system and
the research body dedicated to the domain of Continuous and Implicit Authentication for
mobile security.
Keywords: Machine Learning, Generative Models, Continuous Authentication, Implicit
Authentication, Artificial Intelligence


## More 

Continauth is based on the work of the Buech in 2019. REF: https://github.com/dynobo/ContinAuth
HMOG Dataset is public and can be downloaded from public sources. REF: http://www.cs.wm.edu/~qyang/hmog.html
Home hosts files needed to run this project on Slurm managed resources like the super computers used in Compute Canada, and the Niagara super computer used by us in this work. 
Continauth hosts src/data and src/utility which have scripts to download/generate data and helper code, respectively. 
Continauth also hosts notebooks which have all the jupytner nootbooks that have the experiments. Some results that satisfy the size limitations imposed by github are also hosted under the output directory. 
    │



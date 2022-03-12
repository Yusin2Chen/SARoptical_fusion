# SARoptical Applications
## SAR-Optical Feature Fusion
The effective combination of the complementary information provided by huge amount of unlabeled multi-sensor data (e.g., Synthetic Aperture Radar (SAR) and optical images) is a critical issue in remote sensing. Recently, contrastive learning methods have reached remarkable success in obtaining meaningful feature representations from multi-view data. However, these methods only focus on image-level features, which may not satisfy the requirement for dense prediction tasks such as land-cover mapping. In this work, we propose a self-supervised framework for SAR-optical data fusion and land-cover mapping tasks. SAR and optical images are fused by using a multi-view contrastive loss at image-level and super-pixel level according to one of those possible strategies: in the early, intermediate and late strategies. For the land-cover mapping task, we assign each pixel a land-cover class by the joint use of pre-trained features and spectral information of the image itself. Experimental results show that the proposed approach not only achieves a comparable accuracy but also reduces the dimension of features with respect to the image-level contrastive learning method. Among three fusion strategies, the intermediate fusion strategy achieves the best performance.
### Methods
![Image text](https://github.com/yusin2it/SARoptical_fusion/blob/main/img_sources/Method.jpg)
### Results
![Image text](https://github.com/yusin2it/SARoptical_fusion/blob/main/img_sources/comparison_of_methods.jpg)
### Training
python train_XX.py
### Further Improvements
We further used the multi-crops (multi-scale superpixels) and EMA to stablize the training processes.
python train_twins2s2_shift_spix.py

## Unsupervised SAR-Optical Segmentation
SAR and optical images provide complementary information on land-cover categories in terms of both spectral signatures and dielectric properties. This paper proposes a new unsupervised land-cover segmentation approach based on contrastive learning and vector quantization that jointly uses SAR and optical images. This approach exploits a pseudo-Siamese network to extract and discriminate features of different categories, where one branch is a ResUnet and the other branch is a gumble-softmax vector quantizer.
The core idea is to minimize the contrastive loss between the learned features of the two branches. To segment images, for each pixel the output of gumble-softmax is discretized as a one-hot vector and its proxy label is chosen as the corresponding class. The proposed approach is validated on a subset of DFC2020 dataset including six different land-cover categories. Experimental results demonstrate improvements over the current state-of-the-art techniques and the effectiveness of unsupervised land-cover segmentation on SAR-optical image pairs.
### Methods
![Image text](https://github.com/yusin2it/SARoptical_fusion/blob/main/img_sources/proposal_lc.jpg)
### Results
![Image text](https://github.com/yusin2it/SARoptical_fusion/blob/main/img_sources/BSCD.jpg)
### Training
Python train_vq_Efusion.py

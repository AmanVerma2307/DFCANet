# Learning Joint Local-Global Iris Representations via Spatial Calibration for Generalized Presentation Attack Detection
***
## Description
This is the code repository for the paper "Learning Joint Local-Global Iris Representations via Spatial Calibration for Generalized Presentation Attack Detection" accepted in the *IEEE Transactions on Biometrics, Behaviour, and Identity Science* The paper is available at [here](https://ieeexplore.ieee.org/abstract/document/10401986).
## Abstract
Existing Iris Presentation Attack Detection (IPAD) systems do not generalize well across datasets, sensors and subjects. The main reason for the same is the presence of similarities in bonafide samples and attacks, and intricate iris textures. The proposed DFCANet (Dense Feature Calibration Attention-Assisted Network) uses feature calibration convolution and residual learning to generate domain-specific iris feature representations at local and global scales. DFCANet’s channel attention enables the use of discriminative feature learning across channels. Compared to state-of-the-art methods, DFCANet achieves significant performance gains for the IIITD-CLI, IIITD-WVU, IIIT-CSD, Clarkson-15, Clarkson-17, NDCLD-13, and NDCLD-15 benchmark datasets. Incremental learning in DFCANet overcomes data scarcity issues and cross-domain challenges. This paper also pursues the challenging soft-lens attack scenarios. An additional study conducted over contact lens detection task suggests high domain-specific feature modeling capacities of the proposed network.

![alt text](https://github.com/AmanVerma2307/DFCANet/blob/main/docs/IRPAD_DFCANet.png)

## Enviroment
```
python: 3.9.5
tensorflow: 2.5.0/2.6.0
```

```
Use the following:
conda install -c pkgs/main tensorflow=2.6.0=gpu_py39he88c5ba_0
```

## Trained models
We will release the trained models soon.

## Citing this repository
If you find this code useful in your research, please consider citing us:

```
@article{jaswal2024learning,
  title={Learning Joint Local-Global Iris Representations via Spatial Calibration for Generalized Presentation Attack Detection},
  author={Gaurav Jaswal and Aman Verma and Sumantra Dutta Roy and Raghavendra Ramachandra},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2024},
  publisher={IEEE}
}
```

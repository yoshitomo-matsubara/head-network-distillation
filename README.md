# Head Network Distillation for Split Computing

The official implementations of Head Network Distillation (HND) studies for image classification tasks:
- "Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems," [IEEE Access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639)  
[PDF (Open Access)]  
- "Distilled Split Deep Neural Networks for Edge-assisted Real-time Systems," [MobiCom 2019 Workshop HotEdge '19](https://www.microsoft.com/en-us/research/event/the-1st-workshop-on-hot-topics-in-video-analytics-and-intelligent-edges/)  
[[PDF (Open Access)](https://dl.acm.org/doi/abs/10.1145/3349614.3356022)]

![HND for Split Computing](img/hnd_split_computing.png) 

## Citations
```bibtex
@article{matsubara2020neural,
  title={Head Network Distillation: Splitting Distilled Deep Neural Networks for Resource-constrained Edge Computing Systems},
  author={Matsubara, Yoshitomo and Callegaro, Davide and Baidya, Sabur and Levorato, Marco and Singh, Sameer},
  year={2020}
}

@inproceedings{matsubara2019distilled,
  title={Distilled Split Deep Neural Networks for Edge-assisted Real-time Systems},
  author={Matsubara, Yoshitomo and Baidya, Sabur and Callegaro, Davide and Levorato, Marco and Singh, Sameer},
  booktitle={Proceedings of the 2019 Workshop on Hot Topics in Video Analytics and Intelligent Edges},
  pages={21--26},
  year={2019}
}
```

## Requirements
- Python 3.6
- pipenv
- [myutils](https://github.com/yoshitomo-matsubara/myutils)


## How to clone
```
git clone https://github.com/yoshitomo-matsubara/head-network-distillation.git
cd head-network-distillation/
git submodule init
git submodule update --recursive --remote
pipenv install
```

## Trained models
We publish bottleneck-injected DenseNet-169, -201, Resnet-152 and Inception-v3 trained on 
ILSVRC 2012 (a.k.a. ImageNet) dataset in the following three methods:
- [Naive training](https://drive.google.com/file/d/1yvFslgeewBsHx_GpSJd1MFEcbVTn_Ymq/view?usp=sharing)
- [Knowledge Distillation](https://drive.google.com/file/d/16Q6KxUXjgK5vCsQ5IGt5P1Z21FVAE54R/view?usp=sharing)
- [Head Network Distillation](https://drive.google.com/file/d/1EpTMxSGMU9tDUpEX3bIj_EXGskzAdZC_/view?usp=sharing)

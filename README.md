# Memory Attention Networks (MANs)

Skeleton-based action recognition task is entangled with complex spatio-temporal variations of skeleton joints, and remains challenging for Recurrent Neural Networks (RNNs). In this work, we propose a temporal-then-spatial recalibration scheme to alleviate such complex variations, resulting in an end-to-end Memory Attention Networks (MANs) which consist of a Temporal Attention Recalibration Module (TARM) and a Spatio-Temporal Convolution Module (STCM). Specifically, the TARM is deployed in a residual learning module that employs a novel attention learning network to recalibrate the temporal attention of frames in a skeleton sequence. The STCM treats the attention calibrated skeleton joint sequences as images and leverages the Convolution Neural Networks (CNNs) to further model the spatialand temporal information of skeleton data. These two modules (TARM and STCM) seamlessly form a single network architecture that can be trained in an end-to-end fashion. MANs significantly boost the performance of skeleton-based action recognition and achieve the best results on four challenging benchmark datasets: NTU RGB+D, HDM05, SYSU-3D and UT-Kinect.

![The architecture of MANs](https://github.com/memory-attention-networks/MANs/tree/master/image/architecture.jpeg)

We provide a demo for MANs on [NTU RGB+D dataset](http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp) based on Keras.


## Install
To run this demo, you should install these dependencies:  

    Keras 2.0.8
    tensorflow 1.3.0

## Run demo 
    python process_data.py
    python train_MANs.py

The experimental results of the comparison algorithm are directly quoted from the corresponding papers.

## Citation
if you find MANs useful in your research, please consider citing:  

    @inproceedings{Xie2018Memory,
      title={Memory Attention Networks for Skeleton-based Action Recognition},
      author={Xie, Chunyu and Li, Ce and Zhang, Baochang and Chen, Chen and Han, Jungong and Zou, Changqing and Liu, Jianzhuang},
      booktitle={International Joint Conference on Artificial Intelligence},
      pages={1639--1645},
      year={2018},
    }

The paper link: [Memory Attention Networks for Skeleton-based Action Recognition](https://www.researchgate.net/publication/324717512_Memory_Attention_Networks_for_Skeleton-based_Action_Recognition)

## Contact

yuxie@buaa.edu.cn
Any discussions, suggestions and questions are welcome!


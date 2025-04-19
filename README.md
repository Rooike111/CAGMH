# Covariance Attention Guidance Mamba Hashing for cross-modal retrieval [Paper](https://www.sciencedirect.com/science/article/pii/S0952197625007778) 

This paper is accepted for [Engineering Applications of Artificial Intelligence](https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence) (EAAI).
If you have any questions please contact [wangg@stu.xju.edu.cn](mailto:wangg@stu.xju.edu.cn) .

url:https://github.com/Rooike111/CAGMH

## Dependencies
We use python to build our code, you need to install those package to run

- pytorch 1.12.1
- sklearn
- tqdm
- pillow

## Training

### Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), [nuswide](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)(include all), mirflickr25k [Baidu, 提取码:u9e1](https://pan.baidu.com/s/1upgnBNNVfBzMiIET9zPfZQ) or [Google drive](https://drive.google.com/file/d/18oGgziSwhRzKlAjbqNZfj-HuYzbxWYTh/view?usp=sharing) (include mirflickr25k and mirflickr25k_annotations_v080),  [IAPR TC-12](https://www.kaggle.com/datasets/parhamsalar/iaprtc12 )
then use the "data/make_XXX.py" to generate .mat file

After all mat file generated, the dir of `dataset` will like this:
~~~
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
├── IAPR TC-12
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Notice! It is a txt file!
    ├── index.mat 
    └── label.mat
~~~

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start

After the dataset has been prepared, we could run the follow command to train.
> python main.py --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --numclass 80

# Dataset

The dataset files we have made can be used at the following link:

baiduPan: https://pan.baidu.com/s/12t8tJK0mLfz8A6ocV5ii9g passward：CAGM

# Result

Our hash results are shown below：

baiduPan: https://pan.baidu.com/s/1vLtNMNy_mqAOlz_79gO60w passward：CAGM

## Citation

@article{WANG2025110777,
title = {Covariance Attention Guidance Mamba Hashing for cross-modal retrieval},

journal = {Engineering Applications of Artificial Intelligence},

volume = {152},

pages = {110777},

year = {2025},

issn = {0952-1976},

doi = {https://doi.org/10.1016/j.engappai.2025.110777},

url = {https://www.sciencedirect.com/science/article/pii/S0952197625007778},

author = {Gang Wang and Shuli Cheng and Anyu Du and Qiang Zou},

keywords = {Artificial intelligence, Multimedia technology, Cross-modal hashing, Multi-feature fusion, Covariance attention},

}


## Acknowledegements
[DCHMT](https://github.com/kalenforn/DCHMT), [DSPH](https://github.com/QinLab-WFU/DSPH)

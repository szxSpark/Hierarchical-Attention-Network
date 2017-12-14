# Hierarchical-Attention-Network
HAN model. Three versions.
- **version 1**:  Tensorflow, dynamic GRU
- **version 2**:  Pytorch, pack_padded_sequence
- **version 3**:  Pytorch, mask matrix
# Paper
[Hierarchical Attention Network (readed in 2017/10 by szx)](http://www.aclweb.org/anthology/N16-1174)
# Task Instruction
- **Text Classification**
# Code Instruction
- **data**:  API for loading text data

- **models**:  HAN model

  - hierarchical_tf.py : version 1
  - hierarchical_pack.py : version 2
  - hierarchical_mask.py : version 3
  
- **preprocessor**:  Data preprocessing

- **utils**:  tools for training

- **config.py**:  hyperparameters for models
# Run code
```
CUDA_VISIBLE_DEVICES=X python train.py --model-id 4 --is-save (y or n)
```
# Write in the end
The model finished in BDCI2017.
We implement eight models, such as FastText, TextCNN, TextRNN, RCNN, HAN, CNNInception, CNNWithDoc2Vec, RCNNWithDoc2Vec.
I just make much contribution to HAN model.
Thanks for my teammates, lhliu, dyhu.

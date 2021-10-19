# TIAGE

TIAGE is a benchmark for topic-shift aware dialog modeling. This repo contains the code and data used in [TIAGE: A Benchmark for Topic-Shift Aware Dialog Modeling](https://arxiv.org/abs/2109.04562).

This package is mainly contributed by [Huiyuan Xie](https://huiyuanxie.github.io), [Zhenghao Liu](https://edwardzh.github.io), [Chenyan Xiong](https://www.microsoft.com/en-us/research/people/cxiong/), [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/), [Ann Copestake](https://www.cl.cam.ac.uk/~aac10/). Feel free to [contact](hx255@cl.cam.ac.uk) the authors for questions about our paper. 

## How to cite

If you use TIAGE in your work, please cite:
```
@article{xie2021tiage,
  title={TIAGE: A Benchmark for Topic-Shift Aware Dialog Modeling},
  author={Xie, Huiyuan and Liu, Zhenghao and Xiong, Chenyan and Liu, Zhiyuan and Copestake, Ann},
  journal={arXiv preprint arXiv:2109.04562},
  year={2021}
 }
```


### Install

First clone the repository:
> git clone https://github.com/HuiyuanXie/tiage.git

Install the required packages:
> pip install -r requirements.txt

Note: Please choose appropriate PyTorch version based on your machine (related to your CUDA version). For details, refer to https://pytorch.org/.


## Data

Our TIAGE dataset with human annotated topic-shift labels can be found in the *data* folder. The weak supervision data we used for our experiment is from the train split of the original PersonaChat dataset, which can be downloaded [here](https://github.com/facebookresearch/ParlAI/tree/master/parlai/tasks/personachat).

## Training

Our implementations of BERT, T5 and DialoGPT models are based on the [HuggingFace Transformers library](https://github.com/huggingface/transformers), initialized from their pretrained weights. We use the base version of BERT and T5, and the small version of DialoGPT. To replicate the results in our paper, we recommend to optimise the models using Adam with 5e-5 learning rate and a batch size of 64. We set the maximum input sequence length to 512 to capture as much context information as possible. Our model training is carried out using 1 Nvidia RTX 8000 GPU and takes around 15 hours. 

## Testing

For the topic-shift detection tasks, we report precision, recall and F1-score in our paper. We use the [nlg-eval](https://github.com/Maluuba/nlg-eval) package for automatic evaluation of generated dialog response. 


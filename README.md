# ViTSTR
## Vision Transformer for Fast and Efficient Scene Text Recognition

ViTSTR is a simple single-stage model that uses a pre-trained Vision Transformer (ViT) to perform Scene Text Recognition (ViTSTR).
It has a comparable accuracy with state-of-the-art STR models although it uses significantly less number of parameters and FLOPS.
ViTSTR is also fast due to the parallel computation inherent to ViT architecture. 

### Paper
* [ICDAR 2021](https://link.springer.com/chapter/10.1007/978-3-030-86549-8_21)
* [Arxiv](https://arxiv.org/abs/2105.08582)

![ViTSTR Model](https://github.com/roatienza/deep-text-recognition-benchmark/raw/master/figures/vitstr_model.png)

ViTSTR is built using a fork of [CLOVA AI Deep Text Recognition Benchmark](https://github.com/clovaai/deep-text-recognition-benchmark). Below we document how to train and evaluate ViTSTR-Tiny and ViTSTR-small.

### Install

```
python3 setup.py install
```

### Quick validation using a pre-trained model 

ViTSTR-Small

```
python3 scripts/predict.py
# or
python3 scripts/predict_each.py
```

Available model weights:

| Tiny | Small  | Base |
| :---: | :---: | :---: |
| `vitstr_tiny_patch16_224` | `vitstr_small_patch16_224` | `vitstr_base_patch16_224`|
|[ViTSTR-Tiny](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_tiny_patch16_224.pth)|[ViTSTR-Small](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224.pth)|[ViTSTR-Base](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_base_patch16_224.pth)|
|[ViTSTR-Tiny+Aug](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_tiny_patch16_224_aug.pth)|[ViTSTR-Small+Aug](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_small_patch16_224_aug.pth)|[ViTSTR-Base+Aug](https://github.com/roatienza/deep-text-recognition-benchmark/releases/download/v0.1.0/vitstr_base_patch16_224_aug.pth)|


### Benchmarks (Top 1% accuracy)

| Model | IIIT | SVT | IC03 | IC03 | IC13 | IC13 | IC15 | IC15 | SVTP | CT | Acc | Std
| :--- | :---: | :---: | :---: | :---: | :--: | :--: | :---: | :---: | :---: | :---: | :---: | :--: |
|  | 3000 | 647 | 860 | 867 | 857 |1015 |1811 |2077 |645 |288 |% |  %|
| TRBA (Baseline) | 87.7	|87.4	|94.5	|94.2	|93.4	|92.1	|77.3	|71.6	|78.1	|75.5	|84.3	|0.1
| ViTSTR-Tiny | 83.7 | 83.2 | 92.8 | 92.5 | 90.8 | 89.3 | 72.0 | 66.4 | 74.5 | 65.0 | 80.3| 0.2
| ViTSTR-Tiny+Aug | 85.1	|85.0	|93.4	|93.2	|90.9	|89.7	|74.7	|68.9	|78.3	|74.2	|82.1	|0.1
| ViTSTR-Small | 85.6	|85.3	|93.9	|93.6	|91.7	|90.6	|75.3	|69.5	|78.1	|71.3	|82.6	|0.3
| ViTSTR-Small+Aug  | 86.6	|87.3	|94.2	|94.2	|92.1	|91.2	|77.9	|71.7	|81.4	|77.9	|84.2	|0.1
| ViTSTR-Base  | 86.9	|87.2	|93.8	|93.4	|92.1	|91.3	|76.8	|71.1	|80.0	|74.7	|83.7	|0.1
| ViTSTR-Base+Aug  | 88.4	|87.7	|94.7	|94.3	|93.2	|92.4	|78.5	|72.6	|81.8	|81.3	|85.2	|0.1


### Comparison with other STR models

#### Accuracy vs Number of Parameters

![Acc vs Parameters](https://github.com/roatienza/deep-text-recognition-benchmark/raw/master/scripts/paper/Accuracy_vs_Number_of_Parameters.png)

#### Accuracy vs Speed (2080Ti GPU)
![Acc vs Speed](https://github.com/roatienza/deep-text-recognition-benchmark/raw/master/scripts/paper/Accuracy_vs_Msec_per_Image.png)

#### Accuracy vs FLOPS
![Acc vs FLOPS](https://github.com/roatienza/deep-text-recognition-benchmark/raw/master/scripts/paper/Accuracy_vs_GFLOPS.png)

### Train
```bash
python3 scripts/train.py
```

### Test

ViTSTR-Tiny. Find the path to `best_accuracy.pth` checkpoint file (usually in `saved_models` folder).

```bash
python3 scripts/test.py
```


## Citation
If you find this work useful, please cite:

```
@inproceedings{atienza2021vision,
  title={Vision transformer for fast and efficient scene text recognition},
  author={Atienza, Rowel},
  booktitle={International Conference on Document Analysis and Recognition},
  pages={319--334},
  year={2021},
  organization={Springer}
}
```


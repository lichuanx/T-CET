
# T-CET:Exploiting Network Compressibility and Topology in Zero-Cost NAS

Here is the official implementation for T-CET zero-cost metrics that searched on ZenNAS search space. Most framework adopted from [ZenNAS](https://github.com/idstcv/ZenNAS)

## Abstract
Neural Architecture Search (NAS) has been widely used to discover high-performance neural network architectures over manually designed approaches. Despite their success, current NAS approaches often require extensive evaluation of many candidate architectures in the search space or training of large super networks. To reduce the search cost, recently proposed zero-cost proxies are utilized to efficiently predict the performance of an architecture. However, while many new proxies have been proposed in recent years, relatively little attention has been dedicated to pushing our understanding of the existing ones, with their mutual effects on each other being a particularly -- but not entirely -- overlooked topic. Contrary to that trend, in our work, we argue that it is worth revisiting and analysing the existing proxies in order to further push the boundaries of zero-cost NAS. Towards that goal, we propose to view the existing proxies through a common lens of network compressibility, trainability, and expressivity, as discussed in pruning literature. Notably, doing so allows us to build a better understanding of the high-level relationship between different proxies as well as refine some of them into their more informative variants.
We leverage these insights to design a novel saliency and metric aggregation method informed by compressibility, orthogonality and network topology. We show that our proposed methods are simple but powerful and yield some state-of-the-art results across popular NAS benchmarks.

## Compare to Other Zero-Shot NAS Proxies on CIFAR-10

We use the ResNet-like search space and search for models within parameter budget 1M. All models are searched by the same evolutionary strategy, trained on CIFAR-10/100 for 1440 epochs with auto-augmentation, cosine learning rate decay, weight decay 5e-4. We report the top-1 accuracies in the following table:


| proxy     | CIFAR-10 | CIFAR-100 | 
|-----------|----------| ----------|
| T-CET(SNIP)   | **97.22%**    | **80.4%**|
| T-CET(Synflow)   | **96.6%**    | **80.0%**|
|Zico| *97.0%* | *80.2%*|
| [Zen-NAS](https://arxiv.org/abs/2102.01063)   | 96.2%    | 80.1%|
| FLOPs     | 93.1%    | 64.7%|
| grad-norm | 92.8%    | 65.4%|
| [synflow](https://arxiv.org/abs/2101.08134)   | 95.1%    | 75.9%| 
| [TE-NAS](https://arxiv.org/abs/2102.11535)    | 96.1%    | 77.2%|
| [NASWOT](https://arxiv.org/abs/2006.04647)    | 96.0%    | 77.5%|
| Random    | 93.5%    | 71.1%|

Please check our paper for more details.


## Reproduce Paper Experiments

### System Requirements

* PyTorch >= 1.5, Python >= 3.7
* You can refering requirements.txt for system setting, please note the package version could slightly different based on you GPU driver version.
* By default, ImageNet dataset is stored under \~/data/imagenet; CIFAR-10/CIFAR-100 is stored under \~/data/pytorch\_cifar10 or \~/data/pytorch\_cifar100

You can refer this command for create enviroment
```
conda env create --name tcet -f environment.yml
conda activate tcet
```

### Install Neccessary Package
Install Zero-cost-nas

```
cd zero-cost-nas
pip install .
```


### Searching on CIFAR-10/100
Searching for CIFAR-10/100 models with budget params < 1M , using T-CET

```bash
scripts/TCET_snip_NAS_cifar_params1M.sh
scripts/TCET_synflow_NAS_cifar_params1M.sh
```

### Searching on ImageNet

Searching for ImageNet models with FLOPs budget from 400M to 800M with T-CET:
``` bash
scripts/TCET_snip_NAS_IM_flops1G.sh
scripts/TCET_snip_NAS_IM_flops600M.sh
scripts/TCET_snip_NAS_IM_flops450M.sh
```

## Scoring Network From Other Search Spaces
We provide a more efficient toolkit repository to score networks from diverse search spaces. [Toolkit](https://github.com/iViolinSolo/zero-cost-proxies)


## Citation

```
Xiang, Lichuan, Hunter, Rosco, Dudziak, Łukasz, Xu, Minghao and Wen, Hongkai (2023) Exploiting network compressibility and topology in zero-cost NAS. In: International Conference on Automated Machine Learning (AutoML 2023), Potsdam/Berlin, Germany, 12–15 Sep 2023 (In Press)
```

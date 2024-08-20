# FuzzyCS

 ![License: GPL v2](https://camo.githubusercontent.com/1b537d3212c421e0362b9c7168f1febd83941d79e8ccd8487309a4a759f7da11/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4c6963656e73652d47504c5f76322d626c75652e737667) 

------

### FuzzyCS: Fuzzy Client Selection in Heterogeneous Federated Learning

# Abstract

------

The client selection of federated learning presents a pivotal dilemma necessitating the harmonization of performance and efficiency.
Prevailing studies frame this dilemma within the context of a multi-armed bandit optimization problem. The task grows exponentially more complex with an expanding client base, necessitating advanced algorithms to navigate the decision space efficiently.
Fuzzy sets offer a clear depiction of the uncertainties associated with client selection, eliminating the need for intricate numerical computations.
 To this end, this paper proposes a fuzzy client selection (FuzzyCS) method. 
Specifically, we construct a fuzzy multi-criteria information matrix based on several selection criteria and use it to introduce a criteria-oriented fuzzy selection model. By leveraging the fuzzy rough approximations, we can estimate the fuzzy relationship between the criteria-oriented fuzzy concepts of each client and its likelihood of being selected. FuzzyCS can actively balance performance and efficiency by constructing an adaptive fuzzy approximation set, which incorporates the upper and lower approximate fuzzy rough sets.
The effectiveness of FuzzyCS is rigorously determined through comprehensive comparative experiments, encompassing 11 baseline strategies across 3 benchmark datasets.
The result shows that FuzzyCS has the highest overall ranking in terms of both performance and efficiency by using the Friedman test, indicating that FuzzyCS can effectively achieve the intricate equilibrium between performance and efficiency. The code is available at  https://github.com/dqlme/FuzzyCS.

------

# Introduction

- This is the official PyTorch implementation paper submitted to aaai2025.

- We provide a  plug-and-play platform for federated client selection.

- This platform is consisted of the following content:

- ```
  |---clients
  |     |---client.py
  |     |---model_trainer.py
  |     |---trainer.py
  |     |---trainerbase.py
  |---servers
  |     |---server.py
  |---datas
  |     |---data_utils.py
  |     |---load_data.py
  |     |---load_femnist.py
  |     |---load_harbox.py
  |     |---load_shakespeare.py
  |---models
  |     |---cnn.py
  |     |---lstm.py
  |     |---resnet.py
  |---selection
  |     |---AFL.py
  |     |---Powd.py
  |     |---delta.py
  |     |---divfl.py
  |     |---mcdm.py
  |     |---selection_base.py
  |---log
  |---main.py
  |---options.py
  |---run.sh
  ​```
  ```

# Implementation details

## Requirements

1. **library packages**

   ```
   torch
   logging
   math
   numpy
   PIL
   torchvision
   h5py
   sklearn
   ```

   

2. **download datasets**

   **Shakespeare**

   ```
   wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/shakespeare.tar.bz2
   tar -xvf shakespeare.tar.bz2
   ```

   **FEMNIST**

   ```
   wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2
   
   ```

   **HARBox**

   ```
   https://github.com/xmouyang/FL-Datasets-for-HAR/tree/main/datasets/HARBox
   ```

## Running

- [ ]   **use shell to run the experiments.**

```
sh run.sh
```

- [ ] **Shakespeare-FuzzyCS**

```
python main.py -m MCDM -c shakespeare.yaml
```

- [ ] **FEMNIST-FuzzyCS**

```
python main.py -m MCDM -c femnist.yaml
```

- [ ] **HARBox-FuzzyCS**

```
python main.py -m MCDM -c harbox.yaml
```

# Citation

If this code is useful for your research, please consider citing in your work:

```
Title={FuzzyCS: Fuzzy Client Selection in Heterogeneous Federated Learning}
Author={Dang, Qianlong and Yang Shuai and Linlin Xie and Qiqi Liu and Tao Zhan }
Year={2024}
Month={Aug}
```

 
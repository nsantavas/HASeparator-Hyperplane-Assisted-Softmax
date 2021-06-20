# HASeparator: Hyperplane-Assisted Softmax

TensorFlow implementation of [HASeparator: Hyperplane-Assisted Softmax](https://ieeexplore.ieee.org/document/9356294/metrics#metrics) 

<p align="center">
<image src="images/circle.png" width="420px" height="400px" />
</p>
</p>

### Abstract
----
Efficient feature learning with Convolutional Neural Networks (CNNs) constitutes an increasingly imperative property since several challenging tasks of computer vision tend to require cascade schemes and modalities fusion. Feature learning aims at CNN models capable of extracting embeddings, exhibiting high discrimination among the different classes, as well as intra-class compactness. In this paper, a novel approach is introduced that has separator, which focuses on an effective hyperplane-based segregation of the classes instead of the common class centers separation scheme. Accordingly, an innovatory separator, namely the Hyperplane-Assisted Softmax separator (HASeparator), is proposed that demonstrates superior discrimination capabilities, as evaluated on popular image classification benchmarks.

### Run experiments
---
You can experiments by running
```
python run.py
```
You can find the HASeparator layer in the [customLayer.py](https://github.com/nsantavas/HASeparator-Hyperplane-Assisted-Softmax/blob/develop/customLayer.py) file. The parameters and the example is executed using the CIFAR-10 Dataset. You can easily change the parameters in the file [params.py](https://github.com/nsantavas/HASeparator-Hyperplane-Assisted-Softmax/blob/develop/params.py)
</p>


```bibtex
@INPROCEEDINGS{9356294,
  author={Kansizoglou, Ioannis and Santavas, Nicholas and Bampis, Loukas and Gasteratos, Antonios},
  booktitle={2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA)}, 
  title={HASeparator: Hyperplane-Assisted Softmax}, 
  year={2020},
  volume={},
  number={},
  pages={519-526},
  doi={10.1109/ICMLA51294.2020.00087}}
```
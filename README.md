# Transfer-Learning-CNN
CNN - Multiple Architectures [Training on CPU(multiple cores) &amp; GPU(CUDA) enabled] (Keras+TensorFlow)

Introduction
-----

This project was started after working on a custom CNN architecture on the bacterial colony classifier research. The idea of Transfer learning or CNN is not new but the dataset and problem statement was introduced a few years ago by a Polish joint research (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0184554).

Since CNN is a deep learning approach there is no pre-processing done on the original dataset. Since I was a student this code was created to support CPU & GPU based training (as per available resources). Changing the # cores can help you maximize hardware allocation while training on big models, if a GPU is not at hand. Whereas if you are using a NVIDIA based GPU which supports cuda then it runs on a GPU. 

このプロジェクトは、細菌コロニー分類器の研究でカスタムCNNアーキテクチャに取り組んだ後に開始されました。 転移学習またはCNNのアイデアは新しいものではありませんが、データセットと問題の説明は、ポーランドの共同研究（https://journals.plos.org/plosone/article?id=10.1371/journal.pone）によって数年前に導入されました。 0184554）。

CNNは深層学習アプローチであるため、元のデータセットに対して前処理は行われません。 私が学生だったので、このコードはCPUとGPUベースのトレーニングをサポートするために作成されました（利用可能なリソースに従って）。 ＃コアを変更すると、GPUが手元にない場合に、大きなモデルでトレーニングしながらハードウェア割り当てを最大化するのに役立ちます。 一方、cudaをサポートするNVIDIAベースのGPUを使用している場合は、GPUで実行されます。


Dataset
-----

This dataset is provided here: http://misztal.edu.pl/software/databases/dibas/
Citation: B. Zieliński, A. Plichta, K. Misztal, P. Spurek, M. Brzychczy-Włoch, and D. Ochońska 
Deep learning approach to bacterial colony classification
PLOS ONE, 12(9), 1-14, 2017


Aim
-----

Goal is to make transfer learning easy and accessible for all in a manner where experimentation is made easier. Feel free to contribute.

Setup
-----

* Download & install Python version==3.8.2 (virtual env recommended)
* pip install tuner_requirements.txt
* split dataset (train/validation) and give its path in main.py after cloning repo
* give path for figures in main.py
* mention appropriate optimizer(s), activation function(s), learning rate(s), #core(s), epoch(s), imageDimensions
* go to main.py and pass appropiate parameters to run code

*NOTE: This setup does not cover NVIDIA GPU (CUDA/cuDNN) installation. Follow tensorflow and keras compatibilty and your hardware specs to install appropriate libraries.


Output
-----

Training + validation will occur & learning rate(s), activation function(s), optimizer(s) will be aggregated in a visual graph.


Issues
-----

Library/python versions may hinder compatibilty to execute the code smoothly. 





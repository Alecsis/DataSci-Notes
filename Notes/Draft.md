# Draft

## Ex3 : Colored MNIST

Par défaut :

CrossEntropy
epochs 10
batchs 10
LR 0.01
SGD 

CrossEntropy
epochs 10
batchs 16
LR 0.003
Adamn
-> Loss 4.0497
-> valset  7.96%
-> testset 8.02%

CrossEntropy
epochs 20
batchs 16
LR 0.03
SGD
-> Loss 0.0180
-> valset  98.55%
-> testset 98.81%

## Ex2 : MNIST

MNIST Shape : [60000, 28, 28]
60 000 images de 28*28 = 784 pixels

Modèle par défaut : 88.11%

i - 10, 1, 28, 28
1 - 10, 50, 26, 26
La convolution a réduit de 2 la taille de l'image
p - 10, 50, 13, 13
Le pooling a divisé par 2 la taille de l'image
2 - 10, 50, 11, 11 
p - 10, 50, 5, 5 

epochs 20
batchs 16
LR 0.03
SGD 
-> Loss 1.40
-> valset  94.70%
-> testset 94.42%

---

J'enlève Softmax sur la dernière couche car il paraît que nn.CrossEntropyLoss dispose déjà d'une foncton d'activation

epochs 20
batchs 16
LR 0.03
SGD 
-> Loss 0.0049
-> valset  98.86%
-> testset 99.14%
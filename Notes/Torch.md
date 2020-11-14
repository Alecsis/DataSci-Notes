# Torch

## Imports communs

```py
# Général
import torch                                        # root package
from torch.utils.data import Dataset, Dataloader    # dataset representation and loading

# Neural Network API
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.nn.functional as F           # layers, activations and more
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace       # hybrid frontend decorator and tracing jit
```

## Fonctionnement d'un réseau

La procédure de formation typique pour un réseau de neurones est la suivante :

- Définir le réseau de neurones qui possède certains paramètres (ou poids) pouvant être appris
- Itération sur un ensemble de données d'entrées
- Processus de contribution par le biais du réseau
- Calculer la perte (dans quelle mesure le résultat est correct)
- Propager les gradients dans les paramètres du réseau
- Mettre à jour les poids du réseau, en utilisant généralement une règle de mise à jour simple : `poids = poids - learning_rate * gradient`

## Exemple de réseau

```py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

La fonction `backwards()` est déjà définie avec `autograd`.

Attention, nn fonctionne toujours avec des batcsh. Par exemple, nn.Conv2d prendra en compte un tenseur 4D de `nÉchantillons x nCanaux x Hauteur x Largeur`. Si vous avez un seul échantillon, il suffit d'utiliser `input.unsqueeze(0)` pour ajouter une fausse dimension de lot.

## Loss function

Une fonction de perte prend la paire d'entrées (sortie, cible) et calcule une valeur qui estime la distance entre la sortie et la cible.

Il existe plusieurs fonctions de perte différentes dans le paquet nn . Une fonction de perte simple est : nn.MSELoss qui calcule l'erreur quadratique moyenne entre l'entrée et la cible.

```py
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

## Backprop

Pour rétropropager l'erreur, il suffit de `loss.backward()`. Vous devez cependant effacer les gradients existants, sinon les gradients s'accumuleront aux gradients existants.

```py
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

## Mise à jour des poids

Soit par la formule `weight = weight - learning_rate * gradient` :

```py
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

Soit par les classes de `torch.optim` :

```py
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```
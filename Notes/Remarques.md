# Remarques

## Variation du Learning rate

```py
nepochs x lr -> loss
20 x 3 -> 0.25
20 x 1 -> 0.17
20 x 0.3 -> 0.0004
20 x 0.1 -> 0.0012
20 x 0.03 -> 0.041
20 x 0.01 -> 0.23
```

Un lr plus petit est plus précis mais nécessite plus de epochs
? Ça tend peut-être à overfit

## Variation du batch size

LR de 0.1

```py
100 : ça prend du temps à converger (20 epochs suffisent pas)
30 : ça converge à 25 epochs
10 : ça converge rapidement (10 epochs)
5 : trop instable, ça semble converger à 10 mais ça déconne après
```

Trop grand = trop long à converger
Trop petit = trop instable

## Variation de l'optimize

Sur 20 epochs

```py
SGD x 0.1 -> 100% super
Adam x 0.1 -> 50% trop nul
RMSprop x 0.1 -> 50% pue la merde

SGD -> 62.5%
Adam x 0.01 -> 90.40%
RMSprop x 0.01 -> 99.80%

SGD -> cf au dessus
Adam x 0.003 -> 99.90%
RMSprop x 0.003 -> 99.90%
```

SGD : bien à 0.1
Adam et RMSprop : 10e-3 ou 10e-4

## Variation de la loss

En classif binaire on utilse BCE ou BCEWithLogitsLoss (pour cette dernière il faut enlever la sigmoide de la dernière couche du réseau)

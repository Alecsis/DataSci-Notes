# Fresh install

```psh
choco install python --version=3.8.6
pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy jupyterlab matplotlib pandas scikit-learn
```

CUDA permet de programmer des GPU en C. Elle est développée par Nvidia, initialement pour ses cartes graphiques GeForce 8 Series, et utilise un pilote unifié utilisant une technique de streaming (flux continu).

PyTorch permet d'effectuer les calculs tensoriels nécessaires notamment pour l'apprentissage profond (deep learning). Ces calculs sont optimisés et effectués soit par le processeur (CPU) soit, lorsque c'est possible, par un processeur graphique (GPU) supportant CUDA.
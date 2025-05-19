# 🧠 Accélération GPU d'un réseau de neurones — Projet GPGPU Centrale Nantes

## 📚 Contexte académique

Ce projet a été réalisé dans le cadre du cours **Programmation sur GPU (GPGPU)** à **Centrale Nantes**, sous la supervision de **Pierre-Emmanuel Hladik**.

L'objectif est de mettre en pratique les concepts de parallélisme des données et de programmation CUDA, en optimisant l'entraînement d'un réseau de neurones artificiel (ANN) initialement écrit en C pour CPU.

## 🚀 Objectif du projet

Optimiser le temps d'entraînement d’un réseau de neurones dense (fully connected feedforward neural network) sur le dataset MNIST en passant d’une version séquentielle sur CPU à une version accélérée sur GPU.

## 🧩 Structure du projet

Un **code de base** écrit en C est fourni et comprend :
- `main.c` : boucle d'entraînement principale, gestion des epochs, évaluation de la précision
- `ann.c` : structure et gestion du réseau de neurones (création, propagation avant/arrière)
- `matrix.c` : opérations matricielles (produit matriciel, Hadamard, transposition, etc.)
- `mnist.c` : lecture du dataset MNIST au format binaire IDX
- `TP.pdf` : énoncé complet du micro-projet (version 1.0.1)

## 🔍 Tâches réalisées

- 🔬 **Profilage** du code séquentiel pour identifier les fonctions les plus coûteuses
- ♻️ **Portage GPU** de certaines opérations critiques (ex : `matrix_dot`, `matrix_function`)
- 🧪 **Comparaison des performances** CPU vs GPU pour différentes tailles de minibatch
- 📊 **Analyse** des accélérations obtenues et des limites de parallélisation rencontrées

## 🖼️ Cas d'utilisation : MNIST

- 60 000 images pour l'entraînement, 10 000 pour le test
- Chaque image est une matrice 28x28 normalisée entre 0 et 1
- Sortie : un vecteur de 10 neurones (représentant les digits 0 à 9)
- Architecture testée par défaut : `[784, 30, 10]`

## 🧑‍🏫 Auteur du code de base

> **Pierre-Emmanuel Hladik**
> Enseignant-Chercheur à l'École Centrale de Nantes  
> pierre-emmanuel.hladik@ls2n.fr 

## 👨‍🎓 Réalisé par

Benjamin Lepourtois  
Apprenti ingénieur en Systèmes Embarqués Communicants  
École Centrale de Nantes — Promotion 2025

---

📎 Voir le document `TP.pdf` pour plus de détails sur les attentes pédagogiques et les consignes d’évaluation.

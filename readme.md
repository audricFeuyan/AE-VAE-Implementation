# 🎨 Auto-encodeurs vs. Auto-encodeurs variationels (AE vs. VAE)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Une étude comparative de l'**Auto-Encodeur Classique (AE)** et de l'**Auto-Encodeur Variationnel (VAE)** implémentée avec **PyTorch** sur le jeu de données **CIFAR-10**. 

Ce projet explore les différences d'architectures, les métriques d'évaluation (MSE, PSNR, ELBO), ainsi que la structure des espaces latents à travers trois scénarios de test pratiques : la reconstruction d'images, le traitement du bruit et la génération autonome.

---

## 📌 Sommaire

- [Présentation du Projet](#-présentation-du-projet)
- [Architectures & Théorie](#-architectures--théorie)
- [Résultats & Comparaison](#-résultats--comparaison)
- [Structure du Dépôt](#-structure-du-dépôt)
- [Installation & Utilisation](#-installation--utilisation)
- [Technologies Utilisées](#-technologies-utilisées)

---

## 🚀 Présentation du Projet

Les modèles de compression et de génération d'images reposent sur la réduction de dimensionnalité. Ce notebook vise à répondre à une question clé : **Pourquoi un Auto-Encodeur classique ne peut-il pas générer de nouvelles images, alors qu'un VAE y parvient ?**

### Fonctionnalités principales :
* **Entraînement complet de deux modèles :** Un Auto-Encodeur Convolutionnel (AE) et un Auto-Encodeur Variationnel (VAE).
* **Suivi d'apprentissage rigoureux :** Enregistrement de l'historique de la perte de reconstruction (MSE), de la perte totale (ELBO), du PSNR (*Peak Signal-to-Noise Ratio*) et de la Divergence KL.
* **Suite de tests à 3 scénarios :**
  1. **Reconstruction d'images réelles** (Jeu de validation CIFAR-10).
  2. **Réponse au bruit pur** (Hallucination et projection vers le manifold appris).
  3. **Génération autonome** (Échantillonnage stochastique $z \sim \mathcal{N}(0, I)$).

---

## 🧠 Architectures & Théorie

### 1. Auto-Encodeur Classique (AE)
- **Principe :** Compresse une image vers un point fixe $z$ dans un espace latent déterministe.
- **Fonction de Perte :** Erreur Quadratique Moyenne (MSE).
- **Limites :** L'espace latent présente des "trous". L'échantillonnage de points aléatoires produit des images incohérentes.

### 2. Auto-Encodeur Variationnel (VAE)
- **Principe :** Projette une image vers une distribution de probabilité $\mathcal{N}(\mu, \sigma^2)$ dans l'espace latent.
- **Astuce de Reparamétrisation :** Permet la rétropropagation à travers une opération aléatoire via la formule $z = \mu + \epsilon \cdot \sigma$ où $\epsilon \sim \mathcal{N}(0, I)$.
- **Fonction de Perte (ELBO) :**
  $$\text{Perte VAE} = \text{Reconstruction (MSE)} + \beta \times \text{Divergence KL}$$

| Composant | Activation | Rôle Théorique |
| :--- | :--- | :--- |
| **Encodeur** | `LeakyReLU(0.2)` | Évite le problème du *Dying ReLU*, préserve un gradient continu. |
| **Décodeur** | `ReLU` + `Sigmoid` | Favorise la sparsité des filtres et borne la sortie dans $[0, 1]$. |
| **Pooling** | `AdaptiveAvgPool2d((8,8))` | Pont dynamique garantissant une dimension latente fixe. |

---

## 📊 Résultats & Comparaison

### Tableau Synthétique des Performances

| Scénario de Test | Auto-Encodeur Classique (AE) | Auto-Encodeur Variationnel (VAE) |
| :--- | :--- | :--- |
| **1. Reconstruction CIFAR-10** | **Très bonne netteté** (PSNR $\approx 19.0 \text{ dB}$). | **Bonne netteté**, légèrement plus lisse (effet de la régularisation KL). |
| **2. Entrée Bruit Pixel Pur** | Produit des artefacts incohérents ou du bruit résiduel. | **Projection fluide** : « Hallucine » une forme structurée proche de CIFAR-10. |
| **3. Génération Autonome ($z \sim \mathcal{N}(0,I)$)** | **Échec** : Images grises ou formes indéterminées. | **Succès** : Génère de nouvelles images synthétiques cohérentes. |

---

## 📁 Structure du Dépôt

```text
.
├── notebooks/
│   └── AE-VAE-Implementation.ipynb     # Notebook principal contenant le code et les graphiques
├── data/                            # Répertoire de téléchargement du dataset CIFAR-10
├── LICENCE                          # Licence du projet
└── readme.md                        # Documentation du projet
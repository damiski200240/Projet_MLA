# Projet_MLA – Fader Networks sur CelebA

Ce dépôt contient le pipeline complet du projet **MLA**, depuis le **prétraitement du dataset CelebA** jusqu’au **test des modèles Fader Networks pré-entraînés**.  
L’objectif est de produire une **baseline qualitative fiable**, qui servira ensuite de référence pour comparer avec notre propre ré-implémentation et entraînement du modèle.

---

## 1. Installation et environnement

### Version de Python recommandée
- **Python 3.12**

### Installation des dépendances
Installer les librairies nécessaires avec :

```bash
pip install torch torchvision pillow numpy tqdm matplotlib
```

---

## 2. Dataset CelebA

### 2.1 Téléchargement
Le dataset **CelebA** n’est pas fourni dans ce dépôt.  
Il doit être téléchargé séparément depuis la source officielle.

### 2.2 Organisation attendue
Après téléchargement, placez le dataset dans le dossier racine du projet :

```
Projet_MLA/
└── CelebA/
    └── CelebA/
        └── images/
            ├── 000001.jpg
            ├── 000002.jpg
            └── ...
```

Le chemin exact attendu par les scripts est :
```
CelebA/CelebA/images/
```

---

## 3. Prétraitement du dataset CelebA

### 3.1 Dossier Data_preprocessed

Après exécution du prétraitement, le dossier suivant est créé :

```
Data_preprocessed/
├── Images_Preprocessed/
└── attributes.pth
```

### 3.2 Pipeline de prétraitement

Le dossier `preprocess/` contient :

- `pretraitement_des_images.py`  
  - crop vertical (20 → 198)  
  - resize 256×256  
  - normalisation des images  

- `pretraitement_des_attributs.py`  
  - lecture de `list_attr_celeba.txt`  
  - sélection des attributs Fader  
  - génération de `attributes.pth`  

- `pretraitement_data.py`  
  - exécute l’ensemble du pipeline  

### 3.3 Lancer le prétraitement
Depuis la racine du projet :

```bash
python preprocess/pretraitement_data.py
```

---

## 4. Tests des modèles pré-entraînés Fader Networks

Cette étape permet de tester les modèles Fader officiels sur les données CelebA prétraitées.  
Elle constitue la **baseline qualitative** du projet.

---

## 5. Principe des Fader Networks

Un **Fader Network** est un auto-encodeur conditionné par un attribut binaire.

- Encodeur : image → espace latent  
- Décodeur : latent + attribut → image modifiée  

Attribut binaire :
- 0 → `[1, 0]`
- 1 → `[0, 1]`

### Interpolations
Les interpolations sont définies par :

```
y(α) = [1 − α , α]
α ∈ [1 − alpha_min , alpha_max]
```

---

## 6. Scripts de test

### 6.1 test_celebA_one.py

Test sur **une seule image**.

Sorties :
```
results/one/<modele>/<img_id>/
├── before.png
├── after.png
├── before_after.png
├── grid.png
└── meta.txt
```

### 6.2 test_celebA_grid5.py

Test sur **cinq images fixes**.

Sorties :
```
results/grid5/<modele>/<ids>/
├── before_after.png
├── interpolations.png
└── meta.txt
```

---

## 7. Commandes d’exemple

### Test ONE
```powershell
python .\tests_pretrained\test_celebA_one.py --model_pth .\models_pre_entraines\eyeglasses.pth --img_id 182638 --alpha_min 2.0 --alpha_max 2.0 --n_interpolations 10

```

```powershell
python .\tests_pretrained\test_celebA_one.py --model_pth .\models_pre_entraines\young.pth --img_id 182638 --alpha_min 10.0 --alpha_max 10.0 --n_interpolations 10

```
```powershell
python .\tests_pretrained\test_celebA_one.py --model_pth .\models_pre_entraines\male.pth --img_id 182638 --alpha_min 10.0 --alpha_max 10.0 --n_interpolations 10


```

### Test GRID5
```powershell
python .\tests_pretrained\test_celebA_grid5.py --model_pth .\models_pre_entraines\eyeglasses.pth --img_ids 182638 190012 195555 200001 202599 --alpha_min 2.0 --alpha_max 2.0 --n_interpolations 12

```

```powershell
python .\tests_pretrained\test_celebA_grid5.py --model_pth .\models_pre_entraines\male.pth --img_ids 182638 190012 195555 200001 202599 --alpha_min 2.0 --alpha_max 2.0 --n_interpolations 10

```
```powershell
python .\tests_pretrained\test_celebA_grid5.py --model_pth .\models_pre_entraines\young.pth --img_ids 182638 190012 195555 200001 202599 --alpha_min 10.0 --alpha_max 10.0 --n_interpolations 10
```


---

## 8. Choix des paramètres alpha

| Modèle | Attribut | alpha_min | alpha_max |
|------|---------|-----------|-----------|
| eyeglasses.pth | Eyeglasses | 1.2 | 1.0 |
| male.pth | Male | 2.0 | 2.0 |
| young.pth | Young | 10.0 | 10.0 |
| narrow_eyes.pth | Narrow Eyes | 10.0 | 10.0 |
| pointy_nose.pth | Pointy Nose | 10.0 | 10.0 |

---

## 9. Objectif final

Les résultats générés servent de baseline qualitative pour comparer avec notre propre implémentation des Fader Networks.



Toutes les commandes doivent être exécutées depuis la **racine du projet `Projet_MLA`**.

---

## Script de test utilisé

### `test_celebA_grid5_trained.py`
### `test_celebA_one_trained.py`
### `test_external_one_trained.py`

Ces scripts permetent de tester le modèle Fader Network **entraîné par notre équipe** sur un ensemble
fixe de 5 images CelebA issues du split test.

 

## Paramètres d’interpolation

Les paramètres principaux sont :
- `alpha_min`
- `alpha_max`
- `n_interpolations`

Les valeurs suivantes ont été retenues pour garantir un effet visuel clair :

| Attribut   | alpha_min | alpha_max |
|------------|-----------|-----------|
| Male       | 2.0       | 2.0       |
| Eyeglasses | 2.0       | 2.0       |
| Young      | 10.0      | 10.0      |

Ces valeurs sont identiques à celles utilisées pour les modèles pré-entraînés,
afin de permettre une comparaison directe.

---

## Commandes d’exécution — Modèles entraînés (par exemple) 

### Test sur 5 images CelebA (GRID5)

#### Attribut **Male**

```powershell
python .\test_modeles_entraines\test_celebA_grid5_trained.py --model_pth modeles_entraines\male.pth --img_ids 202577 202583 202595 202505 202567 --alpha_min 2 --alpha_max 2 --n_interpolations 10

```

```powershell
python .\test_modeles_entraines\test_celebA_grid5_trained.py --model_pth modeles_entraines\eyeglasses.pth --img_ids 202577 202583 202595 202505 202567 --alpha_min 2 --alpha_max 2 --n_interpolations 10


```

```powershell
python .\test_modeles_entraines\test_celebA_grid5_trained.py --model_pth modeles_entraines\young.pth --img_ids 202577 202583 202595 202505 202567 --alpha_min 10 --alpha_max 10 --n_interpolations 10


```





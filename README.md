# 🎬 Analyse de Sentiments - Critiques de Films Allociné

Projet d'analyse de sentiments sur le dataset Allociné utilisant des modèles classiques de Machine Learning et le modèle CamemBERT (Deep Learning).

## 📋 Description

Ce projet implémente une analyse complète de sentiments sur des critiques de films en français. Il comprend :

- **Prétraitement de données** : nettoyage, lemmatisation avec SpaCy
- **Analyse exploratoire (EDA)** : word clouds, distributions, statistiques
- **Modèles classiques** : Naive Bayes, SVM, Random Forest, Régression Logistique
- **Modèle Deep Learning** : Fine-tuning de CamemBERT
- **Évaluation et comparaison** des performances

## 🚀 Installation

### Prérequis

- Python 3.8+
- 8GB RAM minimum (pour l'entraînement de BERT)
- Git

### Configuration de l'environnement

1. **Cloner le repository**
```bash
git clone <votre-repo-url>
cd EDA
```

2. **Créer un environnement virtuel**
```bash
python -m venv .venv
```

3. **Activer l'environnement virtuel**

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

5. **Télécharger le modèle SpaCy français**
```bash
python -m spacy download fr_core_news_sm
```

6. **Installer le kernel Jupyter**
```bash
python -m ipykernel install --user --name=eda_venv --display-name="Python (EDA)"
```

## 📓 Utilisation

### Lancer Jupyter Notebook

```bash
jupyter notebook
```

Ensuite :
1. Ouvrir `sentiment_analysis.ipynb`
2. Sélectionner le kernel **"Python (EDA)"** dans le menu
3. Exécuter les cellules dans l'ordre

### Structure du Notebook

1. **Collecte des données** : Chargement du dataset Allociné (200,000 avis)
2. **Prétraitement** : Nettoyage et lemmatisation
3. **EDA** : Visualisations et statistiques
4. **Modélisation classique** : Entraînement et évaluation
5. **CamemBERT** : Fine-tuning et comparaison
6. **Interprétation** : Analyse des résultats

## 📊 Dataset

- **Source** : Hugging Face Datasets - `allocine`
- **Taille** : 200,000 critiques de films
- **Classes** : Binaire (positif/négatif)
- **Langue** : Français

Le dataset est automatiquement téléchargé lors de la première exécution.

## 🔧 Configuration CPU/GPU

Le notebook détecte automatiquement si CUDA est disponible :

- **GPU disponible** : Utilise le GPU pour l'entraînement BERT (plus rapide)
- **CPU uniquement** : Réduit automatiquement la taille du dataset (200 avis pour la démo)

Pour modifier la taille du dataset sur CPU, ajustez la variable `sample_size` dans la cellule correspondante.

## 📈 Résultats Attendus

Les modèles classiques atteignent généralement :
- **Naive Bayes** : ~85-88% d'accuracy
- **SVM** : ~88-91% d'accuracy
- **Random Forest** : ~85-88% d'accuracy
- **Régression Logistique** : ~88-90% d'accuracy

CamemBERT peut atteindre :
- **CamemBERT fine-tuné** : ~92-95% d'accuracy (avec GPU et dataset complet)

## 🛠️ Technologies Utilisées

- **Python 3.x**
- **Pandas & NumPy** : Manipulation de données
- **Scikit-learn** : Modèles ML classiques
- **SpaCy** : Lemmatisation française
- **Transformers (Hugging Face)** : CamemBERT
- **PyTorch** : Deep Learning
- **Matplotlib & Seaborn** : Visualisations
- **WordCloud** : Nuages de mots

## 📁 Structure du Projet

```
EDA/
│
├── .venv/                      # Environnement virtuel
├── sentiment_analysis.ipynb    # Notebook principal
├── allocine_raw.csv           # Dataset (généré automatiquement)
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
└── .gitignore                # Fichiers à ignorer par Git
```

## ⚠️ Notes Importantes

### Entraînement sur CPU

Si vous utilisez uniquement le CPU :
- Le dataset est réduit à 200 avis pour CamemBERT
- L'entraînement prendra ~2-5 minutes par époque
- Pour un dataset complet, utilisez un GPU ou réduisez `num_train_epochs`

### Gestion de la Mémoire

Pour éviter les problèmes de mémoire :
- Fermez les autres applications
- Réduisez `sample_size` si nécessaire
- Redémarrez le kernel Jupyter entre les expérimentations

## 🔍 Dépannage

### Erreur "df is not defined"
- Exécutez d'abord toutes les cellules d'import et de chargement de données

### Erreur "No module named 'accelerate'"
```bash
pip install accelerate>=0.26.0
```

### Erreur SpaCy
```bash
python -m spacy download fr_core_news_sm
```

### Kernel non trouvé
```bash
python -m ipykernel install --user --name=eda_venv
```

## 📝 License

Ce projet est sous licence MIT.

## 👤 Auteur

Rayane Ibnatik

## 📚 Références

- [Dataset Allociné](https://huggingface.co/datasets/allocine)
- [CamemBERT](https://huggingface.co/camembert-base)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [SpaCy](https://spacy.io/)

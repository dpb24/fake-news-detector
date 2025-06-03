# 📰 Fake News Detector: Binary Classification Model <br>

**Libraries:** `scikit-learn`, `XGBoost`, `matplotlib`, `pandas`, `numpy` <br>
**Dataset:** [ISOT Fake News Detection Dataset](https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/) <br>

In this project we use the 🐍 Python libraries [scikit-learn](https://scikit-learn.org/stable/) and [XGBoost](https://xgboost.readthedocs.io/en/stable/) to build a machine learning model that classifies news articles as fake or real. We combine classical machine learning techniques with engineered textual features to improve model generalisability and performance. <br>

## 🧠 Approach
 - **Text vectorisation:** Bag of Words (BoW)
 - **Feature engineering:** % of special characters & % of capitalised characters
 - **Baseline model:** `DecisionTreeClassifier` with `GridSearchCV`
 - **Ensemble model:** `XGBClassifier` with `RandomizedSearchCV`
 - **Robustness:** Removed dataset-specific artefacts (eg. *reuters*) from BoW to improve generalisability <br>

## ✅ Results
 - 🤖 **XGBoost ensemble** achieved **~99.8%** accuracy, precision, recall, and F1 score
 - **Top feature:** `headline_capitalised` (engineered)
 - **Fun insight:** second most important vectorized word for classification — "Trump" 🇺🇸

## 🔭 Future Work
 - Test on more diverse, real-world datasets
 - Experiment with advanced text vectorisation (eg. word embeddings, transformer models)
 - Compare with alternative classifiers (eg. Support Vector Machines) <br>

📖 Jupyter Notebook: [GitHub](https://github.com/dpb24/fake-news-detector/blob/main/notebooks/Fake_News_Detector.ipynb) | [CoLab](https://colab.research.google.com/drive/1WacZBouhz3WlujSIORFhSaVje6W5upGZ?usp=sharing) | [Kaggle](https://www.kaggle.com/code/davidpbriggs/fake-news-detector) <br>

<p align="center">
    <img src="visuals/distribution of uppercase characters.png" width="800"/>
    <img src="visuals/xgboost - feature importance.png" width="800"/>
    <img src="visuals/xgboost - confusion matrix.png" width="800"/>
</p>

# 🐧 Penguin Body Mass Predictor

> A machine learning web app that predicts a penguin's body mass from its species, island, sex, and body measurements. Built with Streamlit and a scikit-learn pipeline trained on the Palmer Penguins dataset.

**🔗 Live app:** https://your-app-name.streamlit.app

![App screenshot](docs/screenshot.png)

---

## Project summary

This project demonstrates the end-to-end workflow of taking a trained ML model from a notebook and turning it into a deployable web application. Anyone with the URL can enter penguin measurements and instantly get a body mass prediction — no Python or data science background required.

**Problem:** Given physical measurements of a penguin, predict its body mass in grams.
**Approach:** Random Forest regression on the Palmer Penguins dataset (344 penguins, 3 species, 3 islands).
**Result:** Mean Absolute Error of ~280 grams on the held-out test set (R² ≈ 0.79).

---

## Tech stack

| Layer | Tool |
|---|---|
| Model | scikit-learn RandomForestRegressor |
| Preprocessing | scikit-learn ColumnTransformer + Pipeline |
| Serialization | joblib |
| Web app | Streamlit |
| Hosting | Streamlit Community Cloud |
| Version control | Git + GitHub |

---

## Project structure

```
penguin-predictor/
├── app.py                       # Streamlit app — the entry point
├── model/
│   └── penguin_model.joblib     # Trained pipeline (preprocessing + model)
├── notebooks/
│   └── train_model.ipynb        # The notebook that produced the model
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── .gitignore
```

---

## How to run locally

If you want to run the app on your own machine instead of using the deployed version:

```bash
# 1. Clone the repo
git clone https://github.com/your-username/penguin-predictor.git
cd penguin-predictor

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## How the model was built

The full training process is in `notebooks/train_model.ipynb`. Key steps:

1. **Load** the Palmer Penguins dataset via seaborn
2. **Build a pipeline** that combines preprocessing and the model into a single object:
   - Numeric features (bill length, bill depth, flipper length): median imputation + standardization
   - Categorical features (species, island, sex): most-frequent imputation + one-hot encoding
   - Model: RandomForestRegressor with 100 trees
3. **Train** on an 80/20 train/test split
4. **Save** the entire pipeline as a single `.joblib` file

> **Why save the whole pipeline?** Saving just the model means you have to recreate the encoding logic in the Streamlit app, which is error-prone. Saving the pipeline means the app receives raw user input (text categories, raw numbers) and the pipeline handles everything internally.

---

## Model performance

| Metric | Value |
|---|---|
| Mean Absolute Error (test) | ~280 g |
| R² (test) | 0.79 |
| Training rows | 274 |
| Test rows | 69 |

The strongest predictors of body mass are flipper length and species — Gentoo penguins are noticeably heavier than Adelie or Chinstrap.

---

## License

MIT — feel free to fork and adapt.

---

*Built as part of the CVERSE Data Science Cohort, Week 11.*
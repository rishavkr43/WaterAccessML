
# WaterAccessML

Predict whether a household has access to an **Improved Source of Drinking Water** using India’s NSSO MIS-78 survey data.

This repository contains code exported from IBM watsonx.ai AutoAI, specifically a Jupyter Notebook implementing a LightGBM-based machine learning model and all necessary assets to reproduce predictions and model evaluation.

---

## Contents

- `notebook/water_access_autoai.ipynb` : Exported AutoAI notebook (core code)
- `data/` : Raw and processed datasets (MIS-78 survey)
- `models/` : Saved (pickled) trained model pipelines
- `requirements.txt` : Python dependencies
- `README.md` : Project documentation

---

## Quick Start

1. **Clone the repository**
git clone https://github.com/your-user/WaterAccessML.git
cd WaterAccessML

text
2. **Prepare your environment**
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt

text
3. **Launch the notebook**
jupyter notebook notebook/water_access_autoai.ipynb

text
The first cells will load and reconstruct the AutoAI pipeline and LightGBM model.

4. **Test prediction sample**
from joblib import load
import pandas as pd

model = load("models/water_access_lgbm.pkl")
sample = pd.read_csv("data/sample_households.csv")
preds = model.predict(sample)
print(preds)

text

---

## Model Details

| Item                     | Value                                    |
|--------------------------|------------------------------------------|
| Algorithm                | LightGBM Classifier (AutoAI Pipeline #2) |
| Cross-validated accuracy | 0.991                                    |
| Input features           | 25 socio-economic & infra variables      |
| Target label             | `Improved_Source_Water` (Yes/No)         |
| Training method          | 3-fold CV, hyperparameter optimization   |

---

## Retraining

1. Update or add new survey data to the `data/` folder.
2. Rerun notebook training cells.
3. Updated model pipeline will be saved to `models/`.

---

## Deployment (Optional)

Users with IBM Cloud access:
- Create a **watsonx.ai Runtime** (Lite plan is sufficient).
- Promote the trained model from `models/` to a Deployment Space.
- Deploy as an **Online Service** for REST prediction API.

---

## Project Status

- [x] Data analysis and cleaning
- [x] Model selection with AutoAI
- [x] AutoAI notebook export and refactor
- [ ] SHAP/interpretability dashboard (planned)
- [ ] Web/app user interface (planned)
- [ ] CI/CD workflows (planned)

---

## Contributing

PRs (pull requests) are welcome.  
Start a discussion before major changes or enhancements.

---

## License

MIT License. See `LICENSE`.

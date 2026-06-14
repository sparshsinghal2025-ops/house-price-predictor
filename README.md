# 🏡 House Price Prediction Dashboard

A production-grade Machine Learning web application that predicts real estate valuations based on key property characteristics. This end-to-end system includes automated data preprocessing, multi-model evaluation, and a fully interactive cloud interface.

🔗 **Live Application URL:** [View Live App](https://streamlit.app)

---

## 🚀 Key Features

* **Interactive Form Interface:** Responsive multi-column input layout matching real-world housing specifications.
* **Automated Processing Pipeline:** On-the-fly handling of categorical encoding, scaling, and array matrix structural formatting.
* **Valuation Visual Distribution:** Real-time generation of market distribution charts to contextualize calculated pricing metrics.
* **Optimized Server Layout:** Lightweight deployment footprint utilizing zero-version-lock configurations.

---

## 🛠️ System Architecture & Tech Stack

* **Frontend Dashboard:** Streamlit
* **Core Machine Learning:** Scikit-Learn, NumPy, Pandas
* **Model Serialization:** Pickle (Comprehensive Bundle Deployment)
* **Statistical Visualization:** Matplotlib, Seaborn
* **Host Platform:** GitHub, Streamlit Community Cloud

---

## 📊 Performance & Optimization Analysis

During development, four distinct regression algorithms were evaluated using the R² (Coefficient of Determination) and Root Mean Squared Error (RMSE) performance metrics:

1. **Linear Regression (Winner):** Highest overall accuracy ($R^2 \approx 0.715$), confirming strong linear feature relationships.
2. **Ridge Regression:** ($R^2 \approx 0.713$) Robust handling of multi-collinear attributes.
3. **XGBoost Regressor:** ($R^2 \approx 0.683$) Advanced tree boosting.
4. **Random Forest Regressor:** ($R^2 \approx 0.657$) Standard ensemble bagging.

The system automatically serializes and utilizes the winning **Linear Regression** model for all real-time predictions.

---

## 💻 Local Setup & Execution

To run this dashboard project on your local computer machine:

1. Clone the repository files:
   ```bash
   git clone https://github.com
   cd house-price-predictor
   ```
2. Install the lightweight clean package requirements list:
   ```bash
   pip install -r requirements.txt
   ```
3. Boot up the Streamlit engine terminal host:
   ```bash
   streamlit run app.py
   ```

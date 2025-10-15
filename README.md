
---

## 🧾 **Dataset**

The dataset represents one month of customer interactions from the **Shopzilla** e-commerce platform.

**Key features include:**
- Channel name, category, and sub-category  
- Customer remarks (feedback text)  
- Order metadata (ID, timestamps)  
- Product and price information  
- Agent-level details (name, shift, tenure)  
- Target: CSAT score (integer satisfaction rating)

📎 **Dataset Link:** [Shopzilla CSAT Dataset](https://drive.google.com/file/d/14IJWsaVX8OXW97M1fsYb-m9CyjuhchpJ/view?usp=sharing)

---

## ⚙️ **Model Development Workflow**

1. **Data Cleaning** – Handle missing values and incorrect timestamps.  
2. **Feature Engineering** – Create new predictive variables like response delay.  
3. **Text Preprocessing** – Clean and vectorize customer remarks.  
4. **Transformation Pipeline** – Encode categorical features, scale numerics, and combine text embeddings.  
5. **Model Training** – ANN with ReLU activations, dropout, and Adam optimizer.  
6. **Evaluation** – Assess with accuracy, F1-score, and MAE.  
7. **Explainability** – Use SHAP for feature importance interpretation.  
8. **Deployment** – Serve model via Streamlit app for real-time predictions.

---

## 🧩 **ANN Architecture Summary**

| Layer | Units | Activation | Notes |
|-------|--------|-------------|-------|
| Input | varies | — | Structured + text features combined |
| Dense (1) | 128 | ReLU | With L2 regularization |
| Dense (2) | 64 | ReLU | Captures complex interactions |
| Dropout | 0.3 | — | Prevents overfitting |
| Output | #classes | Softmax | Predicts CSAT score class |

---

## 📈 **Evaluation Metrics**

| Metric | Purpose |
|--------|----------|
| Accuracy | Overall correctness of predictions |
| Macro F1-score | Balances precision/recall across CSAT classes |
| Mean Absolute Error | Penalizes large ordinal deviations |
| Confusion Matrix | Highlights misclassification patterns |

**Example Insight:**  
- Longer response delay → lower CSAT  
- Positive remarks → higher CSAT  
- Experienced agents and morning shifts → higher satisfaction  

---

## 🧩 **Streamlit Application**

The **CSAT Prediction App** allows non-technical users to interact with the model easily.  
**How it works:**
1. Users input interaction details (channel, remarks, price, shift, etc.).  
2. The preprocessing pipeline transforms the inputs.  
3. The trained ANN predicts the CSAT score.  
4. The predicted score is displayed instantly.

To run the app locally:

```bash
streamlit run app/csat_app.py

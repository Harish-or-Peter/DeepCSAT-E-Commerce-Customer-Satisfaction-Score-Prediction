import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from datetime import datetime


# ---- Paths
# project_path = "/content/drive/MyDrive/AlmaBetter/Specialization_Track/CSAT"
fe_path   = r"artifects\feature_engineering_pipeline.pkl"
model_path= r"artifects\csat_ann.keras"
le_path   =  r"artifects\label_encoder.joblib"

# ---- Load artifacts
preprocessor = joblib.load(fe_path)
model = tf.keras.models.load_model(model_path)
le = joblib.load(le_path)

# ---- Feature preparation (same as training pipeline)
def prepare_features(raw_dict):
    df = pd.DataFrame([raw_dict])

    # response delay
    df["response_delay"] = (
        pd.to_datetime(df["issue responded"]) - pd.to_datetime(df["issue reported at"])
    ).dt.total_seconds() / 3600.0

    # missing indicators
    df["missing_Product_category"] = df["Product_category"].isna().astype(int)
    df["missing_Item_price"] = df["item price"].isna().astype(int)

    # item price binning
    bins = [0, 100, 500, 1000, 5000, np.inf]
    labels = ["0-100", "100-500", "500-1000", "1000-5000", "5000+"]
    df["Item_price_bin"] = pd.cut(df["item price"].fillna(0), bins=bins, labels=labels)

    # final order (must match training)
    final_df = df[
        [
            "channel_name",
            "category",
            "Sub-category",
            "Customer Remarks",
            "Product_category",
            "Tenure Bucket",
            "Agent Shift",
            "Item_price_bin",
            "missing_Product_category",
            "missing_Item_price",
            "response_delay",
        ]
    ]
    return final_df

# ---- Prediction function
def predict_single(input_dict):
    features = prepare_features(input_dict)
    X_t = preprocessor.transform(features).astype("float32")
    probs = model.predict(X_t)
    pred_class = probs.argmax(axis=1)[0]
    pred_csat = le.inverse_transform([pred_class])[0]
    return int(pred_csat)

# ---- Streamlit UI
st.title("📊 CSAT Prediction App")

# ---- Group 1: Channel & Category
col1, col2, col3 = st.columns(3)
with col1:
    channel = st.selectbox("Channel Name", ["Inbond", "Outcall", "Email"])
with col2:
    category = st.selectbox("Category", [
        'Product Queries', 'Order Related', 'Returns', 'Cancellation',
        'Shopzilla Related', 'Payments related', 'Refund Related',
        'Feedback', 'Offers & Cashback', 'Onboarding related', 'Others',
        'App/website'
    ])
with col3:
    subcat = st.selectbox("Sub-category", [
        'Life Insurance', 'Product Specific Information', 'Installation/demo',
        'Reverse Pickup Enquiry', 'Not Needed', 'Fraudulent User',
        'Exchange / Replacement', 'Missing', 'General Enquiry', 'Return request',
        'Delayed', 'Service Centres Related', 'Payment related Queries',
        'Order status enquiry', 'Return cancellation', 'Unable to track',
        'Seller Cancelled Order', 'Wrong', 'Invoice request',
        'Priority delivery', 'Refund Related Issues', 'Signup Issues',
        'Online Payment Issues', 'Technician Visit', 'UnProfessional Behaviour',
        'Damaged', 'Product related Issues', 'Refund Enquiry',
        'Customer Requested Modifications', 'Instant discount', 'Card/EMI',
        'Shopzila Premium Related', 'Account updation', 'COD Refund Details',
        'Seller onboarding', 'Order Verification', 'Other Cashback',
        'Call disconnected', 'Wallet related', 'PayLater related', 'Call back request',
        'Other Account Related Issues', 'App/website Related', 'Affiliate Offers',
        'Issues with Shopzilla App', 'Billing Related', 'Warranty related',
        'Others', 'e-Gift Voucher', 'Shopzilla Rewards', 'Unable to Login',
        'Non Order related', 'Service Center - Service Denial', 'Payment pending',
        'Policy Related', 'Self-Help', 'Commission related'
    ])

# ---- Group 2: Customer Remarks
remarks = st.text_area("Customer Remarks", height=80)

# ---- Group 3: Order info
col1, col2, col3 = st.columns(3)
with col1:
    order_id = st.number_input("Order ID", min_value=1, step=1)
with col2:
    order_date = st.date_input("Order Date Time")
with col3:
    survey_date = st.date_input("Survey Response Date")

col1, col2 = st.columns(2)
with col1:
    issue_reported = st.date_input("Issue Reported At")
with col2:
    issue_responded = st.date_input("Issue Responded")

# ---- Group 4: Product info
col1, col2, col3 = st.columns(3)
with col1:
    product_category = st.selectbox("Product Category", [
        'Unknown', 'LifeStyle', 'Electronics', 'Mobile', 'Home Appliences',
        'Furniture', 'Home', 'Books & General merchandise', 'GiftCard', 'Affiliates'
    ])
with col2:
    item_price = st.number_input("Item Price", min_value=0.0, step=0.1)
with col3:
    handling_time = st.number_input("Connected Handling Time", min_value=0.0, step=0.1)

# ---- Group 5: Agent info
col1, col2 = st.columns(2)
with col1:
    tenure = st.selectbox("Tenure Bucket", ['On Job Training', '>90', '0-30', '31-60', '61-90'])
with col2:
    shift = st.selectbox("Agent Shift", ['Morning', 'Evening', 'Split', 'Afternoon', 'Night'])

# ---- Predict button
if st.button("Predict CSAT"):
    sample = {
        "channel_name": channel,
        "category": category,
        "Sub-category": subcat,
        "Customer Remarks": remarks,
        "Product_category": product_category,
        "Tenure Bucket": tenure,
        "Agent Shift": shift,
        "item price": item_price,
        "order id": order_id,
        "order date time": str(order_date),
        "issue reported at": str(issue_reported),
        "issue responded": str(issue_responded),
        "survey response date": str(survey_date),
        "connected handling time": handling_time,
    }

    csat = predict_single(sample)
    st.success(f"✅ Predicted CSAT Score: {csat}")


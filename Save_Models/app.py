import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

# =========================================================
# Custom Styling (this is what gives the "special" look)
# =========================================================
st.markdown("""
    <style>
    .main {
        background-color: #FFFBF5;
    }
    .hero-box {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 20px rgba(255, 107, 53, 0.25);
    }
    .hero-box h1 {
        margin: 0;
        font-size: 34px;
    }
    .hero-box p {
        margin-top: 6px;
        font-size: 16px;
        opacity: 0.95;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
        font-size: 16px;
    }
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-box">
        <h1>🍔🛵 Smart Food Delivery Time Predictor</h1>
        <p>Explore your delivery data, predict delivery times, and enjoy the experience of trying our smart model! </p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# Load Model (trained WITHOUT a sklearn Pipeline)
# =========================================================
@st.cache_resource
def load_model_artifacts():
    folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(folder, "xgb_model.pkl")
    columns_path = os.path.join(folder, "xgb_features.pkl")

    model = joblib.load(model_path)          # your trained regressor (e.g. XGBoost / RF / etc.)
    feature_columns = joblib.load(columns_path)  # X.columns saved right after get_dummies() in your notebook
    return model, feature_columns

try:
    model, feature_columns = load_model_artifacts()
    model_ready = True
except FileNotFoundError:
    model_ready = False
    st.warning(
        "⚠️ Model files not found next to this script.\n\n"
        "Put these two files in the same folder as this app:\n"
        "- `xgb_model.pkl`  (your trained model, saved with joblib.dump)\n"
        "- `xgb_features.pkl`   (X.columns after your get_dummies step, saved with joblib.dump)"
    )

# =========================================================
# Encoding maps — must match EXACTLY what you used in training
# =========================================================
ORDINAL_MAPS = {
    "Traffic_Level": {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3},
    "Restaurant_Load": {"Low": 0, "Medium": 1, "High": 2},
    "Delivery_Priority": {"Normal": 0, "Priority": 1, "VIP": 2},
}

NOMINAL_COLS = [
    "Day_of_Week", "Weather", "Pickup_Zone",
    "Dropoff_Zone", "Vehicle_Type", "Cuisine_Type"
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEATHERS = ["Clear", "Cloudy", "Fog", "Rain", "Storm"]
ZONES = ["Commercial", "CBD", "Industrial", "Residential", "Suburban"]
VEHICLES = ["Bike", "Electric Scooter", "Scooter", "Car"]
CUISINES = ["Biryani", "Burger", "Cafe", "Chinese", "Desserts",
            "North Indian", "Pizza", "South Indian"]

# =========================================================
# Tabs — Data Explorer  |  Predict
# =========================================================
tab_explore, tab_predict = st.tabs(["📊 Data Explorer", "🔮 Predict Delivery Time"])

# =========================================================
# TAB 1 — Data Explorer
# =========================================================
with tab_explore:
    with st.sidebar:
        st.header("📂 Upload Dataset")
        file = st.file_uploader("Upload your delivery CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        st.subheader("📋 Data Preview")
        if st.checkbox("Show Data"):
            st.dataframe(df, use_container_width=True)

        st.subheader("📊 Quick Info")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("📈 Visualization")
        if len(df.columns) >= 2:
            v1, v2 = st.columns(2)
            with v1:
                x_col = st.selectbox("X-axis", df.columns)
            with v2:
                y_col = st.selectbox("Y-axis", df.columns, index=1)

            chart_type = st.radio("Chart type", ["Bar", "Line", "Scatter"], horizontal=True)

            try:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col, color_discrete_sequence=["#FF6B35"])
                elif chart_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col, color_discrete_sequence=["#FF6B35"])
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, color_discrete_sequence=["#FF6B35"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Cannot create this chart: {e}")

        st.subheader("🔎 Filter Data")
        filter_col = st.selectbox("Column to filter", df.columns, key="filter_col")
        unique_vals = df[filter_col].dropna().unique()
        if len(unique_vals) > 0:
            filter_val = st.selectbox("Value", unique_vals, key="filter_val")
            filtered_df = df[df[filter_col] == filter_val]
            st.write(f"Filtered rows: {len(filtered_df)}")
            st.dataframe(filtered_df, use_container_width=True)

        st.subheader("💬 Ask About Data")
        question = st.text_input("Example: average Time_taken_min")
        if question:
            q = question.lower()
            found_col = None
            for col in df.columns:
                if col.lower() in q:
                    found_col = col
                    break

            if found_col:
                if "average" in q or "mean" in q:
                    if pd.api.types.is_numeric_dtype(df[found_col]):
                        st.info(f"Average {found_col}: {df[found_col].mean():.2f}")
                    else:
                        st.warning("This column is not numeric.")
                elif "max" in q or "highest" in q:
                    st.info(f"Maximum {found_col}: {df[found_col].max()}")
                elif "min" in q or "lowest" in q:
                    st.info(f"Minimum {found_col}: {df[found_col].min()}")
                else:
                    st.info("Try asking about average, mean, max or min.")
            else:
                st.warning("Column not found.")
                st.write(df.columns.tolist())
    else:
        st.info("👈 Upload your delivery dataset from the sidebar to start exploring.")

# =========================================================
# TAB 2 — Prediction
# =========================================================
with tab_predict:
    st.subheader("Enter order details")

    with st.form("delivery_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**📦 Order**")
            order_hour = st.number_input("Order Hour", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", DAYS)
            is_weekend = st.selectbox("Is Weekend?", ["No", "Yes"])
            is_festival = st.selectbox("Is Festival?", ["No", "Yes"])
            order_items = st.number_input("Number of Items", 1, 20, 2)

        with c2:
            st.markdown("**🌦️ Conditions**")
            weather = st.selectbox("Weather", WEATHERS)
            traffic_level = st.selectbox("Traffic Level", ["Low", "Moderate", "High", "Severe"])
            pickup_zone = st.selectbox("Pickup Zone", ZONES)
            dropoff_zone = st.selectbox("Dropoff Zone", ZONES)
            vehicle_type = st.selectbox("Vehicle Type", VEHICLES)

        with c3:
            st.markdown("**👤 Rider & Restaurant**")
            rider_experience = st.number_input("Rider Experience (Years)", 0.0, 20.0, 3.0, 0.1)
            rider_rating = st.number_input("Rider Rating", 1.0, 5.0, 4.0, 0.1)
            restaurant_rating = st.number_input("Restaurant Rating", 1.0, 5.0, 4.0, 0.1)
            restaurant_load = st.selectbox("Restaurant Load", ["Low", "Medium", "High"])
            delivery_priority = st.selectbox("Delivery Priority", ["Normal", "Priority", "VIP"])

        st.markdown("**🛵 Trip Details**")
        d1, d2, d3 = st.columns(3)
        with d1:
            cuisine_type = st.selectbox("Cuisine Type", CUISINES)
        with d2:
            preparation_time = st.number_input("Preparation Time (min)", 1, 100, 20)
            road_distance = st.number_input("Road Distance (km)", 0.1, 100.0, 10.0, 0.1)
        with d3:
            number_of_signals = st.number_input("Number of Signals", 0, 50, 8)
            average_speed = st.number_input("Average Speed (km/h)", 1.0, 100.0, 30.0, 0.1)

        predict_clicked = st.form_submit_button("🚀 Predict Delivery Time", use_container_width=True)

    if predict_clicked:
        if not model_ready:
            st.error("Model isn't loaded. Add `xgb_model.pkl` and `xgb_features.pkl` next to this file first.")
        else:
            try:
                # -------------------------------------------------
                # 1) Build the raw row exactly like your training data
                # -------------------------------------------------
                raw_input = {
                    "Order_Hour": order_hour,
                    "Is_Weekend": 1 if is_weekend == "Yes" else 0,
                    "Is_Festival": 1 if is_festival == "Yes" else 0,
                    "Rider_Experience_Years": rider_experience,
                    "Rider_Rating": rider_rating,
                    "Restaurant_Rating": restaurant_rating,
                    "Order_Items": order_items,
                    "Preparation_Time_Min": preparation_time,
                    "Road_Distance_km": road_distance,
                    "Number_of_Signals": number_of_signals,
                    "Average_Speed_kmph": average_speed,
                    "Traffic_Level": traffic_level,
                    "Restaurant_Load": restaurant_load,
                    "Delivery_Priority": delivery_priority,
                    "Day_of_Week": day_of_week,
                    "Weather": weather,
                    "Pickup_Zone": pickup_zone,
                    "Dropoff_Zone": dropoff_zone,
                    "Vehicle_Type": vehicle_type,
                    "Cuisine_Type": cuisine_type,
                }
                new_order = pd.DataFrame([raw_input])

                # -------------------------------------------------
                # 2) Ordinal encoding (same maps as your notebook)
                # -------------------------------------------------
                for col, mapping in ORDINAL_MAPS.items():
                    new_order[col] = new_order[col].map(mapping)

                # -------------------------------------------------
                # 3) One-hot encoding (same as pd.get_dummies in your notebook)
                # -------------------------------------------------
                new_order = pd.get_dummies(
                    new_order, columns=NOMINAL_COLS, drop_first=True, dtype=int
                )

                # -------------------------------------------------
                # 4) Align columns to exactly match training features
                # -------------------------------------------------
                new_order = new_order.reindex(columns=feature_columns, fill_value=0)
                new_order = new_order.astype(float)

                # -------------------------------------------------
                # 5) Predict directly with the model (no pipeline)
                # -------------------------------------------------
                prediction = model.predict(new_order)[0]
                prediction = max(0, round(float(prediction), 1))

                st.success(f"⏱️ Estimated Delivery Time: **{prediction} minutes**")

                if prediction <= 40:
                    st.info("🟢 Fast delivery expected.")
                elif prediction <= 70:
                    st.warning("🟡 Moderate delivery time expected.")
                else:
                    st.error("🔴 Longer delivery time expected.")

            except Exception as e:
                st.error("An error occurred while making the prediction.")
                st.exception(e)
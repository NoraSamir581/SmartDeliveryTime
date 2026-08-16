import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import joblib
import os

# Page Configuration
# =========================

st.set_page_config(
    page_title="Smart Delivery Time",
    page_icon="🚴",
    layout="wide"
)


# =========================
# Title
# =========================

st.title("🚴 Smart Delivery Time Prediction")

st.write(
    "Analyze delivery data and predict the estimated delivery time."
)


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model_artifacts():

    folder = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(folder, "xgb_model.pkl")
    features_path = os.path.join(folder, "xgb_features.pkl")

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)

    return model, feature_columns

try:

    model, feature_columns = load_model_artifacts()

except FileNotFoundError:

    st.error(
        "Model files not found. Make sure these files are in the same "
        "folder as appy.py:\n\n"
        "- xgb_model.pkl\n"
        "- xgb_features.pkl"
    )

    st.stop()


# =========================
# Upload Data
# =========================

with st.sidebar:

    st.header("📂 Upload Data")

    file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


# =========================
# Data Analysis Section
# =========================

if file:

    df = pd.read_csv(file)


    # -------------------------
    # Data Preview
    # -------------------------

    st.subheader("📋 Data Preview")

    if st.checkbox("Show Data"):

        st.dataframe(
            df,
            use_container_width=True
        )


    # -------------------------
    # Data Information
    # -------------------------

    st.subheader("📊 Data Information")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.write(
        "Columns:",
        df.columns.tolist()
    )


    # =========================
    # Visualization
    # =========================

    st.subheader("📈 Visualization")

    if len(df.columns) >= 2:

        c1, c2 = st.columns(2)

        with c1:

            x_col = st.selectbox(
                "Select X-axis",
                df.columns
            )

        with c2:

            y_col = st.selectbox(
                "Select Y-axis",
                df.columns,
                index=1
            )


        chart_type = st.radio(
            "Choose Chart",
            ["Bar", "Line", "Scatter"],
            horizontal=True
        )


        try:

            if chart_type == "Bar":

                fig = px.bar(
                    df,
                    x=x_col,
                    y=y_col
                )

            elif chart_type == "Line":

                fig = px.line(
                    df,
                    x=x_col,
                    y=y_col
                )

            else:

                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col
                )


            st.plotly_chart(
                fig,
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                f"Cannot create this chart: {e}"
            )


    # =========================
    # Filter Data
    # =========================

    st.subheader("🔎 Filter Data")

    filter_column = st.selectbox(
        "Select column",
        df.columns,
        key="filter_column"
    )

    unique_values = (
        df[filter_column]
        .dropna()
        .unique()
    )


    if len(unique_values) > 0:

        selected_value = st.selectbox(
            "Select value",
            unique_values,
            key="filter_value"
        )

        filtered_df = df[
            df[filter_column] == selected_value
        ]

        st.write(
            f"Filtered rows: {len(filtered_df)}"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True
        )


    # =========================
    # Ask About Data
    # =========================

    st.subheader("💬 Ask About Data")

    question = st.text_input(
        "Example: average Time_taken_min"
    )


    if question:

        question_lower = question.lower()

        selected_column = None


        for column in df.columns:

            if column.lower() in question_lower:

                selected_column = column

                break


        if selected_column:

            if (
                "average" in question_lower
                or "mean" in question_lower
            ):

                if pd.api.types.is_numeric_dtype(
                    df[selected_column]
                ):

                    st.info(
                        f"Average {selected_column}: "
                        f"{df[selected_column].mean():.2f}"
                    )

                else:

                    st.warning(
                        "This column is not numeric."
                    )


            elif (
                "max" in question_lower
                or "highest" in question_lower
            ):

                st.info(
                    f"Maximum {selected_column}: "
                    f"{df[selected_column].max()}"
                )


            elif (
                "min" in question_lower
                or "lowest" in question_lower
            ):

                st.info(
                    f"Minimum {selected_column}: "
                    f"{df[selected_column].min()}"
                )


            else:

                st.info(
                    "Try asking about average, mean, max or min."
                )


        else:

            st.warning(
                "Column not found."
            )

            st.write(
                df.columns.tolist()
            )


else:

    st.info(
        "👈 Upload your delivery dataset from the sidebar "
        "to explore the data."
    )


# =========================================================
# Prediction Section
# =========================================================

st.divider()

st.header("🔮 Predict Delivery Time")

st.write(
    "Enter the details of a new delivery order."
)


# =========================
# Prediction Form
# =========================

with st.form("delivery_prediction_form"):


    # =========================
    # Order Information
    # =========================

    st.subheader("📦 Order Information")

    c1, c2, c3 = st.columns(3)


    with c1:

        order_hour = st.number_input(
            "Order Hour",
            min_value=0,
            max_value=23,
            value=12
        )


        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )


        is_weekend = st.selectbox(
            "Is Weekend?",
            ["No", "Yes"]
        )


    with c2:

        is_festival = st.selectbox(
            "Is Festival?",
            ["No", "Yes"]
        )


        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Cloudy",
                "Fog",
                "Rain",
                "Storm"
            ]
        )


        traffic_level = st.selectbox(
            "Traffic Level",
            [
                "Low",
                "Moderate",
                "High",
                "Severe"
            ]
        )


    with c3:

        pickup_zone = st.selectbox(
            "Pickup Zone",
            [
                "Commercial",
                "CBD",
                "Industrial",
                "Residential",
                "Suburban"
            ]
        )


        dropoff_zone = st.selectbox(
            "Dropoff Zone",
            [
                "Commercial",
                "CBD",
                "Industrial",
                "Residential",
                "Suburban"
            ]
        )


        vehicle_type = st.selectbox(
            "Vehicle Type",
            [
                "Bike",
                "Electric Scooter",
                "Scooter",
                "Car"
            ]
        )


    # =========================
    # Rider & Restaurant
    # =========================

    st.subheader("👤 Rider & Restaurant")

    c1, c2, c3 = st.columns(3)


    with c1:

        rider_experience = st.number_input(
            "Rider Experience (Years)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.1
        )


        rider_rating = st.number_input(
            "Rider Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )


    with c2:

        restaurant_rating = st.number_input(
            "Restaurant Rating",
            min_value=1.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )


        restaurant_load = st.selectbox(
            "Restaurant Load",
            [
                "Low",
                "Medium",
                "High"
            ]
        )


    with c3:

        cuisine_type = st.selectbox(
            "Cuisine Type",
            [
                "Biryani",
                "Burger",
                "Cafe",
                "Chinese",
                "Desserts",
                "North Indian",
                "Pizza",
                "South Indian"
            ]
        )


        delivery_priority = st.selectbox(
            "Delivery Priority",
            [
                "Normal",
                "Priority",
                "VIP"
            ]
        )


    # =========================
    # Delivery Details
    # =========================

    st.subheader("🛵 Delivery Details")

    c1, c2, c3 = st.columns(3)


    with c1:

        order_items = st.number_input(
            "Number of Items",
            min_value=1,
            max_value=20,
            value=2
        )


        preparation_time = st.number_input(
            "Preparation Time (Minutes)",
            min_value=1,
            max_value=100,
            value=20
        )


    with c2:

        road_distance = st.number_input(
            "Road Distance (km)",
            min_value=0.1,
            max_value=100.0,
            value=10.0,
            step=0.1
        )


        number_of_signals = st.number_input(
            "Number of Signals",
            min_value=0,
            max_value=50,
            value=8
        )


    with c3:

        average_speed = st.number_input(
            "Average Speed (km/h)",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=0.1
        )


    # =========================
    # Prediction Button
    # =========================

    predict_clicked = st.form_submit_button(
        "🚀 Predict Delivery Time",
        use_container_width=True
    )


# =========================================================
# Prediction
# =========================================================

if predict_clicked:

    try:

        # -------------------------
        # Feature Engineering
        # -------------------------

        peak_hours = [
            8,
            9,
            10,
            18,
            19,
            20
        ]


        is_peak_hour = int(
            order_hour in peak_hours
        )


        signals_per_km = (
            number_of_signals / road_distance
            if road_distance > 0
            else 0
        )


        prep_time_per_item = (
            preparation_time / order_items
            if order_items > 0
            else 0
        )


        # -------------------------
        # Create Raw Input
        # -------------------------

        raw_input = {

            "Order_Hour": order_hour,

            "Is_Weekend":
                1 if is_weekend == "Yes" else 0,

            "Is_Festival":
                1 if is_festival == "Yes" else 0,

            "Rider_Experience_Years":
                rider_experience,

            "Rider_Rating":
                rider_rating,

            "Restaurant_Rating":
                restaurant_rating,

            "Restaurant_Load":
                restaurant_load,

            "Road_Distance_km":
                road_distance,

            "Traffic_Level":
                traffic_level,

            "Average_Speed_kmph":
                average_speed,

            "Delivery_Priority":
                delivery_priority,

            "Is_Peak_Hour":
                is_peak_hour,

            "Day_of_Week":
                day_of_week,

            "Weather":
                weather,

            "Pickup_Zone":
                pickup_zone,

            "Dropoff_Zone":
                dropoff_zone,

            "Vehicle_Type":
                vehicle_type,

            "Cuisine_Type":
                cuisine_type,

            "Signals_per_km":
                signals_per_km,

            "Prep_Time_per_Item":
                prep_time_per_item,

            "Order_Items":
                order_items,

            "Preparation_Time_min":
                preparation_time,

            "Number_of_Signals":
                number_of_signals
        }


        new_order = pd.DataFrame(
            [raw_input]
        )


        # =========================
        # Ordinal Encoding
        # =========================

        new_order["Traffic_Level"] = (
            new_order["Traffic_Level"]
            .map({
                "Low": 0,
                "Moderate": 1,
                "High": 2,
                "Severe": 3
            })
        )


        new_order["Restaurant_Load"] = (
            new_order["Restaurant_Load"]
            .map({
                "Low": 0,
                "Medium": 1,
                "High": 2
            })
        )


        new_order["Delivery_Priority"] = (
            new_order["Delivery_Priority"]
            .map({
                "Normal": 0,
                "Priority": 1,
                "VIP": 2
            })
        )


        # =========================
        # One Hot Encoding
        # =========================

        nominal_columns = [

            "Day_of_Week",

            "Weather",

            "Pickup_Zone",

            "Dropoff_Zone",

            "Vehicle_Type",

            "Cuisine_Type"
        ]


        new_order = pd.get_dummies(
            new_order,
            columns=nominal_columns,
            drop_first=True,
            dtype=int
        )


        # =========================
        # Match Model Features
        # =========================

        new_order = new_order.reindex(
            columns=feature_columns,
            fill_value=0
        )


        new_order = new_order.astype(float)


        # =========================
        # Prediction
        # =========================

        prediction = model.predict(
            new_order
        )[0]


        prediction = max(
            0,
            round(float(prediction), 1)
        )


        # =========================
        # Display Result
        # =========================

        st.success(
            f"⏱️ Estimated Delivery Time: "
            f"**{prediction} minutes**"
        )


        if prediction <= 40:

            st.info(
                "🟢 Fast delivery is expected."
            )


        elif prediction <= 70:

            st.warning(
                "🟡 Moderate delivery time is expected."
            )


        else:

            st.error(
                "🔴 Longer delivery time is expected."
            )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.exception(e)
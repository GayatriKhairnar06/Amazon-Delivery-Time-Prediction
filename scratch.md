
# 🚚 Amazon Delivery Time Prediction

---

## 🪶 Project Summary

In today’s fast-paced e-commerce environment, **timely delivery** plays a crucial role in customer satisfaction and operational efficiency.  
This project focuses on predicting the **delivery time (in minutes)** for Amazon orders using machine learning techniques.  
By analyzing multiple factors such as **agent age, ratings, distance, traffic, weather, and vehicle type**, the system builds a predictive model that estimates how long it will take for a package to be delivered.

The final model is deployed using a **Streamlit web application**, enabling users to input delivery details and instantly view the predicted time.  
All trained models were **tracked and compared using MLflow**, ensuring reproducibility and transparency throughout the experiment lifecycle.

---

## 💡 Problem Statement

Amazon faces challenges in maintaining **accurate delivery time estimations** due to numerous dynamic factors such as:
- Traffic congestion  
- Weather disturbances  
- Delivery agent performance  
- Distance and route variability  

These unpredictable factors often lead to **delays, reduced customer trust, and higher operational costs**.  
Thus, there is a strong need for a **data-driven model** that can accurately predict delivery times under varying conditions.

---

## 🎯 Business Objective

The main business goal is to:
1. **Predict** the delivery duration of Amazon orders accurately.  
2. **Optimize** delivery routes and agent assignments based on predicted delivery times.  
3. **Improve** customer satisfaction by providing more reliable delivery estimates.  
4. **Enhance** operational efficiency by identifying delay-causing factors (e.g., weather or traffic).  
5. **Enable monitoring and tracking** of model performance using **MLflow**, ensuring continuous improvement.

---

## 📝  Documentation

### 🎯 Objective
To predict the estimated delivery time (in minutes) for Amazon orders based on various real-world features such as location, distance, weather, traffic, and delivery agent performance.

---

### 📊 Dataset Description

| Feature | Description |
|----------|-------------|
| `Order_ID` | Unique identifier for each order |
| `Agent_Age` | Age of the delivery agent |
| `Agent_Rating` | Rating of the delivery agent |
| `Store_Latitude`, `Store_Longitude` | Coordinates of the store location |
| `Drop_Latitude`, `Drop_Longitude` | Coordinates of the delivery location |
| `Weather` | Weather conditions (e.g., Sunny, Fog, Sandstorm, etc.) |
| `Traffic` | Traffic density (Low, Medium, High, Jam) |
| `Vehicle` | Type of vehicle used for delivery |
| `Distance_km` | Calculated distance between store and drop location |
| `Is_Weekend` | Boolean feature indicating if order was placed on weekend |
| `Order_Year`, `Order_Month`, `Order_Day`, `Order_Weekday`, `Order_Hour` | Extracted time-based features |
| `Delivery_Time` | Target variable (actual delivery duration in minutes) |

---

### ⚙️ Implementation Steps

#### 1. Data Preprocessing
- Handled missing values.
- Extracted new time-based features from `Order_DateTime` and `Pickup_DateTime`.
- Encoded categorical columns (`Weather`, `Traffic`, `Vehicle`) using **LabelEncoder**.
- Removed unnecessary columns: `Order_Time`, `Area`, and `Category`.
- Normalized numerical features using **StandardScaler**.

#### 2. Feature Engineering
- Calculated `Distance_km` using the **Haversine formula** for latitude-longitude pairs.
- Added derived features like `Is_Weekend` and `Rush_Hour`.

#### 3. Model Development
Trained multiple regression models:
- Linear Regression   
- Random Forest Regressor   
- Gradient Boosting Regressor  

**Evaluation Metrics:**
- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)

#### 4. Best Model Selection
**Random Forest Regressor** achieved the best results:
- R² Score: **0.81**
- MAE: **17.23 minutes**
- RMSE: **22.52 minutes**

Saved using **Joblib** and integrated into the **Streamlit app** for real-time prediction.

---

### 🌐 Streamlit Application
Developed an interactive dashboard for prediction:
- Users input features like weather, traffic, coordinates, etc.
- Predicts **estimated delivery time** instantly.
- Displays results and visual insights.

---
### 🧾 Conclusion

- The Amazon Delivery Time Prediction system efficiently predicts delivery durations using multiple contextual and environmental parameters.
- By leveraging **MLflow**, all model performances were tracked, compared, and stored systematically.
- The final **Random Forest Model** was deployed through a Streamlit web app on Streamlit Cloud, offering an accurate and interactive real-world solution.


print("************Amazon Delivery Time Prediction******************")
print("********Step 1:- Load DataSet**************")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("amazon_delivery.csv")
print(df.head())
print(df.info())
print(df.describe())
df1 = df.columns.tolist()
print("Original Columns",df1)
print("**********Step 2A :- Handling missing values*********")
missed = df.isnull().sum()
missing_percentage = (missed/len(df))*100
print(pd.DataFrame({'Missig Values': missed,'%Missing Percentage': missing_percentage}))
#Handling missing values when many columns are involved
print("df2 = Case 1 Coulumn is 100% missing")
df2 = df.dropna(axis=1, how='all')
print(df2.head())
df2 = df.columns.tolist()
print("Columns After Removing Missing Values= ",df2)
print("Original Columns",df1)
print("After comparing it is conclude that dataset is having zero column having 100% missing value")
print("Case 2 Column Is mostly missing >50% ")
print("***********Step 2B:- Fill Missing Values************")
print("Agent_rating(Numeric Column)==using median(safer than mean if outliers exist")
df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].median())

print("Weather(Categorical Column)==using mode")
df['Weather'] = df['Weather'].fillna(df['Weather'].mode()[0])
print(df.isnull().sum())

print("Missing values handled successfully!")
print("Remaining Nulls After Filling:\n", df.isnull().sum())
print("***************Step 3 Handling Duplicates**********************8")
duplicates = df.duplicated().sum()
print("Total Duplicates: ",duplicates)
#view duplicate row
print(df[df.duplicated()].head())
#In Case same code used for another dataset containing duplicates
if(duplicates>0):
    duplicate_value_removed = df.drop_duplicates()
    print("Duplicates removed New Shape: ",duplicate_value_removed)
else:
    print("NO duplicates found.")

print("**********Step 4 Feature Engineering*********")
print("Euclidean distance is enough unless exact km/miles are needed.")
import numpy as np
#Euclidean Distance
df['Distance'] = np.sqrt(
    (df['Store_Latitude']-df['Drop_Latitude'])**2 + (df['Store_Longitude'] - df['Drop_Longitude'])**2
)
print(df[['Store_Latitude','Store_Longitude','Drop_Latitude','Distance']].head())
print("===============Histogram of Eucledean distance===============")
plt.figure(figsize=(8,5))
sns.histplot(df['Distance'], bins=50, color='skyblue', kde=True)
plt.title("Distribution of Euclidean Distance")
plt.xlabel("Euclidean Distance")
plt.ylabel("Frequency")
plt.show()

#Haversine
print("Step 4B: Haversine gives a realistic feature instead of a flat approximation."
      "This can improve  model’s accuracy.")
from math import radians, sin, cos, sqrt, atan2
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    curvature_distance = sin(dlat/2)**2 + cos(lat1)* cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(curvature_distance), sqrt(1-curvature_distance))
    R = 6371 # Radius of Earth in km
    return R*c
df['Distance_km'] = df.apply(lambda row: haversine(
    row['Store_Latitude'], row['Store_Longitude'],
    row['Drop_Latitude'], row['Drop_Longitude']),axis=1)
print(df[['Distance_km']].head())
print("=================Histogram of Haversine distances (in km)==================")
plt.figure(figsize=(8,5))
sns.histplot(df['Distance_km'], bins=50, color='orange', kde=True)
plt.title("Distribution of Haversine Distance (km)")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.show()

print("**********Step 5: Time-Based Feature Engineering*********")
# Convert to datetime
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').dt.time
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce').dt.time
# Extract features from Order_Date
df['Order_Year'] = df['Order_Date'].dt.year
df['Order_Month'] = df['Order_Date'].dt.month
df['Order_Day'] = df['Order_Date'].dt.day
df['Order_Weekday'] = df['Order_Date'].dt.weekday  # 0=Monday, 6=Sunday
df['Is_Weekend'] = df['Order_Weekday'].apply(lambda x: 1 if x >= 5 else 0)
# Extract hour from Order_Time and Pickup_Time
df['Order_Hour'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce').apply(lambda x: x.hour if pd.notnull(x) else None)
df['Pickup_Hour'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce').apply(lambda x: x.hour if pd.notnull(x) else None)
# Create Rush Hour Flag (let’s assume 8-10 AM and 5-8 PM are rush hours)
def rush_hour(hour):
    if hour in [8, 9, 10, 17, 18, 19, 20]:
        return 1
    return 0
df['Rush_Hour'] = df['Order_Hour'].apply(lambda x: rush_hour(x) if pd.notnull(x) else 0)
print(df[['Order_Date','Order_Weekday','Is_Weekend','Order_Hour','Pickup_Hour','Rush_Hour']].head())

print("**********Step 6: Pickup Delay Feature*********")
print("Pickup Delay feature measures how long it takes for the agent to actually pick up the parcel after the order is placed.")
# Convert both to datetime (same date for order + pickup time)
df['Order_DateTime'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Order_Time'].astype(str), errors='coerce')
df['Pickup_DateTime'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Pickup_Time'].astype(str), errors='coerce')
# Calculate delay in minutes
df['Pickup_Delay_Minutes'] = (df['Pickup_DateTime'] - df['Order_DateTime']).dt.total_seconds() / 60
# Fill any negative/NaN delays with 0 (sometimes data has issues)
df['Pickup_Delay_Minutes'] = df['Pickup_Delay_Minutes'].apply(lambda x: max(x, 0) if pd.notnull(x) else 0)
print(df[['Order_DateTime','Pickup_DateTime','Pickup_Delay_Minutes']].head())
#to deal with categorical variable
print("*************Step 7: one hot encoding*************")
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Step 7.1: Drop non-predictive or datetime columns
X = df.drop(columns=[
    "Delivery_Time",    # target
    "Order_ID",         # unique identifier
    "Order_Date",       # original date
    "Order_Time",       # original order time
    "Pickup_Time",      # original pickup time
    "Order_DateTime",   # combined datetime
    "Pickup_DateTime"   # combined datetime
])
y = df["Delivery_Time"]
# Step 7.2: List categorical columns
categorical_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]
# Step 7.3: Preprocessing + Model
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
preprocessor = ColumnTransformer(
    transformers=[("cat", categorical_transformer, categorical_cols)],
    remainder="passthrough"
)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])
# Step 7.4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 7.5: Fit model
model.fit(X_train, y_train)
# Step 7.6: Evaluate
print("Train Score:", model.score(X_train, y_train))
print("Test Score:", model.score(X_test, y_test))
#df.to_csv("amazon_processed.csv", index=False)

print("Step 8 :- EDA")
print("Boxplot to check outliers")
plt.figure(figsize=(8,4))
sns.boxplot(y=df['Distance_km'], color='lightgreen')
plt.title("Boxplot of Haversine Distance")
plt.show()
print("Scatter plot vs delivery time")
plt.figure(figsize=(8,5))
sns.scatterplot(x='Distance_km', y='Delivery_Time', data=df, alpha=0.5)
plt.title("Distance vs Delivery Time")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (minutes)")
plt.show()

print("=======Distribution of Delivery_Time (target variable)=========")
plt.figure(figsize=(8,5))
sns.histplot(df['Delivery_Time'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (minutes)")
plt.ylabel("Frequency")
plt.show()

print("==========Numeric Features: Agent_Age, Agent_Rating, Distance_km, Pickup_Delay_Minutes")
numeric_cols = ['Agent_Age','Agent_Rating','Distance_km','Pickup_Delay_Minutes']

plt.figure(figsize=(15,8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2,2,i+1)
    sns.histplot(df[col], bins=30, kde=True, color='orange')
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

print("Boxplots to detect outliers")
plt.figure(figsize=(15,8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2,2,i+1)
    sns.boxplot(y=df[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

print("========Correlation Heatmap===========")
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols + ['Delivery_Time']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("Categorical Features Visualization")
categorical_cols = ["Weather", "Traffic", "Vehicle", "Area", "Category"]

for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette='Set2')
    plt.title(f'Counts of {col}')
    plt.xticks(rotation=45)
    plt.show()
print("Feature Relationships")
plt.figure(figsize=(6,4))
sns.scatterplot(x='Pickup_Delay_Minutes', y='Delivery_Time', data=df)
plt.title("Pickup Delay vs Delivery Time")
plt.show()
'''Benefits of doing this:
See the distribution & skew of numeric variables.
Detect outliers which can affect model performance.
Observe relationships between features and target.
Identify if categorical variables are imbalanced (like Traffic or Weather).
'''
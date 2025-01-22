import numpy as np
import pandas as pd

df = r"C:\Users\GABRU\Downloads\archive (3)\insurance.csv"
data = pd.read_csv(df)



print(data.describe())
print("------------------")
print(data.info())

import seaborn as sns
import matplotlib.pyplot as plt

# Plot correlations
corr_data = data.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_data, annot=True, cmap='coolwarm')
plt.show()



# Create subplots
plt.figure(figsize=(16, 10))

# Plot 'age'
plt.subplot(2, 4, 1)
sns.histplot(data['age'], kde=True, color='blue')
plt.title('Age Distribution')

# Plot 'bmi'
plt.subplot(2, 4, 2)
sns.histplot(data['bmi'], kde=True, color='green')
plt.title('BMI Distribution')

# Plot 'expenses'
plt.subplot(2, 4, 3)
sns.histplot(data['expenses'], kde=True, color='red')
plt.title('Expenses Distribution')

plt.subplot(2, 4, 4)
sns.histplot(data['children'], kde=True, color='red')
plt.title('children Distribution')

plt.subplot(2,4,5)
sns.countplot(x = data['sex'], data = data)
plt.title('Sex data count')

plt.subplot(2,4,6)
sns.countplot(x = data['smoker'],data = data)
plt.title('smoker count')

plt.subplot(2,4,7)
sns.countplot(x = data['region'],data = data)
plt.title('smoker count')
plt.xticks(rotation='vertical')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Identify numerical and categorical columns
numerical_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# Numerical pipeline (impute missing values and scale)
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute with mean
    ('scaler', StandardScaler())  # Scaling features
])

# Categorical pipeline (impute missing values and encode)
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute with 'missing'
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-Hot Encoding
])

# Combine both pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])



preprocessor


from sklearn.model_selection import train_test_split

X = data.drop('expenses', axis=1)
y = data['expenses']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create a pipeline with preprocessor and model
simple_reg = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', LinearRegression())])

simple_reg

from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge Regression
ridge = Pipeline(steps=[('preprocessor', preprocessor),
                         ('model', Ridge(alpha=1))])  # Adjust alpha for regularization strength
ridge

# Lasso
lasso = Pipeline(steps=[('preprocessor', preprocessor),
                         ('model', Lasso(alpha=0.1))])  # Adjust alpha for regularization
lasso

# Elastic Net
elastic_net = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))])  # Adjust hyperparameters
elastic_net

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gb = Pipeline(steps=[('preprocessor', preprocessor),
                     ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))])

gb



# For Simple Linear Regression
simple_reg.fit(X_train, y_train)
y_pred = simple_reg.predict(X_test)

print("Simple Linear Regression R^2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# For Ridge Regularizaton
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge R^2:", r2_score(y_test, y_pred_ridge))

# For Lasso Regularzation
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("Lasso R^2:", r2_score(y_test, y_pred_lasso))

# For EllasticNet
elastic_net.fit(X_train, y_train)
y_pred_en = elastic_net.predict(X_test)
print("ElasticNet R^2:", r2_score(y_test, y_pred_en))


# For Gradient Boosting 
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("Gradient Boosting R^2:", r2_score(y_test, y_pred_gb))



import matplotlib.pyplot as plt

# Assuming you have the following values calculated already:
# For MSE
simple_mse = mean_squared_error(y_test, y_pred)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
elastic_net_mse = mean_squared_error(y_test, y_pred_en)
gb_mse = mean_squared_error(y_test, y_pred_gb)
# xgb_mse = mean_squared_error(y_test, y_pred_xgb)

# For R^2
simple_r2 = r2_score(y_test, y_pred)
ridge_r2 = r2_score(y_test, y_pred_ridge)
lasso_r2 = r2_score(y_test, y_pred_lasso)
elastic_net_r2 = r2_score(y_test, y_pred_en)
gb_r2 = r2_score(y_test, y_pred_gb)
# xgb_r2 = r2_score(y_test, y_pred_xgb)

# Models
models = ['Simple Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'Gradient Boosting']

# MSE values
mse_values = [simple_mse, ridge_mse, lasso_mse, elastic_net_mse, gb_mse]

# R^2 values
r2_values = [simple_r2, ridge_r2, lasso_r2, elastic_net_r2, gb_r2]

plt.figure(figsize=(14, 7))

# Plot MSE
plt.subplot(1, 2, 1)
plt.bar(models, mse_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')
plt.xticks(rotation=45, ha='right')

# Plot R-squared
plt.subplot(1, 2, 2)
plt.bar(models, r2_values, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.title('R-squared (R2)')
plt.ylabel('R2 Score')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()








import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

# Inject custom CSS for colorful background
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ff7e5f,rgb(45, 40, 37));  /* Gradient from pink to orange */
            color: white;  /* Set text color to white for contrast */
        }
        .stButton>button {
            background-color: #ff7e5f;  /* Customize button color */
            color: white;
        }
        .stSelectbox>div>div>input, .stNumberInput>div>div>input {
            background-color: #feb47b;  /* Customize input field color */
            color: black;
        }
        .stTitle {
            color: #fff;
        }
        .stText {
            color: #fff;
        }
    </style>
    """, 
    unsafe_allow_html=True
)


# Load and display an image at the start
image = "C:\\Users\\GABRU\\Downloads\\209949-health-insurance.jpg"
st.image(image, caption="Insurance Premium Prediction", use_column_width=True)

# Assuming your original models (simple_reg, ridge, lasso, elastic_net, gb) are already trained and stored
models = {
    'Simple Linear Regression': simple_reg,  # Replace with your trained model
    'Ridge': ridge,  # Replace with your trained model
    'Lasso': lasso,  # Replace with your trained model
    'ElasticNet': elastic_net,  # Replace with your trained model
    'Gradient Boosting': gb  # Replace with your trained model
}

# Streamlit UI elements for user input
st.title("Insurance Premium Prediction")

# User inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, format="%.2f")
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", ['male', 'female'])
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest', 'southeast', 'northwest', 'northeast'])

# Model selection
model_name = st.selectbox("Select Model", ['Simple Linear Regression', 'Ridge', 'Lasso', 'ElasticNet', 'Gradient Boosting'])

# Prediction logic
if st.button('Predict'):
    model = models.get(model_name)  # Fetch the selected model from the dictionary
    
    if model is not None:
        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'children': [children],
            'sex': [sex],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Make the prediction using the model
        prediction = model.predict(input_data)[0]

        # Display the predicted insurance premium
        st.write(f"Predicted Insurance Premium: ${prediction:,.2f}")
    else:
        st.write("Model not found. Please select a valid model.")



import numpy as np
import pandas as pd
import re
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load datasets
url = "https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"
test_data_with_labels = pd.read_csv(url)

# Load the uploaded test dataset
test_data = pd.read_csv('./input/test.csv')

# Clean up names in both datasets
test_data_with_labels['name'] = test_data_with_labels['name'].apply(lambda x: re.sub('"', '', x))
test_data['Name'] = test_data['Name'].apply(lambda x: re.sub('"', '', x))

# Streamlit app layout
st.title('Titanic Survival Prediction')

# Display dataset
st.subheader('Test Data Preview')
st.write(test_data.head())

# Allow users to select which features to include
st.sidebar.header('Feature Selection')
features = test_data.columns.tolist()

# User-selected features
selected_features = st.sidebar.multiselect('Select features to use for prediction', features)

# Ensure user interaction - if no features are selected, prompt to select at least one feature
if len(selected_features) == 0:
    st.sidebar.warning("Please select at least one feature to proceed.")
else:
    # Prepare data for modeling
    X = test_data[selected_features]

    # Handle non-numeric columns using get_dummies (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # Handle missing values (fill NaNs with the mean value of the column)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Prepare the target variable (survived) based on matching names
    survived = []
    for name in test_data['Name']:
        matched_row = test_data_with_labels.loc[test_data_with_labels['name'] == name]
        if not matched_row.empty:
            survived.append(int(matched_row['survived'].values[-1]))
        else:
            survived.append(0)  # Assign 0 if no matching row found

    y = survived

    # Train a Linear Regression model based on selected features
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions on the full test data
    predictions = model.predict(X)
    predictions = [1 if pred > 0.5 else 0 for pred in predictions]

    # Display the updated prediction results
    st.subheader('Prediction Results Based on Selected Features')
    prediction_df = pd.DataFrame({'Name': test_data['Name'], 'Survived': predictions})
    st.write(prediction_df.head())

    # Allow the user to download the predictions
    submission_file = 'submission.csv'
    prediction_df.to_csv(submission_file, index=False)

    # Provide a download link
    st.subheader('Download Prediction Results')
    st.download_button('Download CSV', submission_file)

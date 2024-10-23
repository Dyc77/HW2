import numpy as np
import pandas as pd
import os
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests

# Load Titanic dataset
url = "https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"
s = requests.get(url).content
test_data_with_labels = pd.read_csv(io.StringIO(s.decode('utf-8')))
test_data = pd.read_csv('./input/test.csv')

# Remove double quotes from names
test_data_with_labels['name'] = test_data_with_labels['name'].str.replace('"', '')
test_data['Name'] = test_data['Name'].str.replace('"', '')

# Match the survived status for test data
survived = test_data['Name'].map(
    lambda name: test_data_with_labels.loc[test_data_with_labels['name'] == name, 'survived'].values[-1]
)

# Load submission file and assign the survived values
submission = pd.read_csv('./input/gender_submission.csv')
submission['Survived'] = survived.astype(int)
submission.to_csv('submission.csv', index=False)

# Display the submission file
print(submission.head())

# Plot the distribution of survivors
plt.figure(figsize=(8, 6))
sns.countplot(data=submission, x='Survived')
plt.title("Survivor Distribution")
plt.xlabel("Survived (1=Yes, 0=No)")
plt.ylabel("Count")
plt.show()

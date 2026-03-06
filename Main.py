from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

training_data = pd.read_csv('Dataset - Final KNN Dataset.csv')
training_labels = training_data.columns.tolist()

# feature: music, target: performance
x = training_data[['Is_Calming_Music', 'Difficulty', 'Response_Time']]
y = training_data['Answered_Correctly']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.9, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train_scaled, y_train)

y_pred = knn.predict(x_test_scaled)


# graphing the prediction
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

correct = training_data[training_data['Answered_Correctly'] == 1]
incorrect = training_data[training_data['Answered_Correctly'] == 0]

ax.scatter(correct['Difficulty'], 
           correct['Response_Time'], 
           correct['Is_Calming_Music'], 
           c='blue', marker='o', label='Correct Response', alpha=0.5)

ax.scatter(incorrect['Difficulty'], 
           incorrect['Response_Time'], 
           incorrect['Is_Calming_Music'], 
           c='red', marker='x', label='Incorrect Response', alpha=0.8)

ax.set_xlabel('Task Difficulty')
ax.set_ylabel('Response Time (ms)')
ax.set_zlabel('Type of Music')
ax.set_title('Music, Task Difficulty, and Response Time vs. Performance')

ax.set_zticks([0, 1])
ax.set_zticklabels(['Non-Calming', 'Calming'])

ax.set_xticks([1, 3])
ax.set_xticklabels(['One-back', 'Three-back'])

ax.legend()

ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# feature matrix to determine which has the most impact on performance

correlations = training_data.corr()['Answered_Correctly'].drop(['Answered_Correctly', 'Correct_Response', 'Actual_Response']).abs().sort_values(ascending=True)

plt.figure(figsize=(13, 6))
colors = ['#ff7f0e' if x < 0.05 else '#1f77b4' for x in correlations.values]

bars = plt.barh(correlations.index, correlations.values, color=colors)

plt.xlabel('Importance (Absolute Correlation Coefficient)')
plt.title('Feature Importance: What Drives Task Performance?')
plt.grid(axis='x', linestyle='--', alpha=0.7)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', va='center')

plt.tight_layout()
plt.show()
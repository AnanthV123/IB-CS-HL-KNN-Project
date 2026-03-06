from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

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

# 2. Create the 3D Figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 3. Separate the data for coloring (Correct vs Incorrect)
# We use the actual data to see how the 'neighborhoods' are formed
correct = training_data[training_data['Answered_Correctly'] == 1]
incorrect = training_data[training_data['Answered_Correctly'] == 0]

# 4. Plot the points
# Floor 0 will be Non-Calming, Floor 1 will be Calming
ax.scatter(correct['Difficulty'], 
           correct['Response_Time'], 
           correct['Is_Calming_Music'], 
           c='blue', marker='o', label='Correct Response', alpha=0.5)

ax.scatter(incorrect['Difficulty'], 
           incorrect['Response_Time'], 
           incorrect['Is_Calming_Music'], 
           c='red', marker='x', label='Incorrect Response', alpha=0.8)

# 5. Labels and Styling
ax.set_xlabel('Task Difficulty')
ax.set_ylabel('Response Time (ms)')
ax.set_zlabel('Is Calming Music (0=No, 1=Yes)')
ax.set_title('3D Feature Space: Music Impact vs. Task Difficulty')

# Set the Z-axis ticks to only show 0 and 1
ax.set_zticks([0, 1])
ax.set_zticklabels(['Non-Calming', 'Calming'])

ax.legend()

# 6. Adjust the viewing angle (optional)
# This angle usually gives the best view of the two 'floors'
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# results = pd.DataFrame({
#     'Music': x_test['Is_Calming_Music'],
#     'Predicted': y_pred
# })

# non_calming_counts = [
#     len(results[(results['Music'] == 0) & (results['Predicted'] == 0)]),
#     len(results[(results['Music'] == 0) & (results['Predicted'] == 1)])
# ]
# calming_counts = [
#     len(results[(results['Music'] == 1) & (results['Predicted'] == 0)]),
#     len(results[(results['Music'] == 1) & (results['Predicted'] == 1)])
# ]

# # 6. Plotting with Matplotlib
# labels = ['Incorrect', 'Correct']
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots(figsize=(10, 6))

# # Create side-by-side bars
# rects1 = ax.bar(x - width/2, non_calming_counts, width, label='Non-Calming', color='#ff7f0e')
# rects2 = ax.bar(x + width/2, calming_counts, width, label='Calming', color='#1f77b4')

# # Add text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Number of Trials')
# ax.set_title('KNN Predictions: Impact of Music on Accuracy')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# # Add value labels on top of bars
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

# fig.tight_layout()
# plt.show()
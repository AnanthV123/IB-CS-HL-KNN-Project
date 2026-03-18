import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Dataset.csv")

data["Letter"] = data["Letter_Ascii"].apply(chr)

X = data[["Letter_Ascii"]]

y = data["Answered_Correctly"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

accuracies = []
k_values = range(1,10)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.figure()

plt.plot(k_values, accuracies, marker='o')

plt.title("KNN Accuracy vs K Value")
plt.xlabel("K (Number of Neighbors)")
plt.ylabel("Prediction Accuracy")

plt.show()


best_k = k_values[accuracies.index(max(accuracies))]
best_accuracy = max(accuracies)

print("Best K:", best_k)
print("Prediction Accuracy:", best_accuracy)


data["Letter"] = data["Letter_Ascii"].apply(chr)

letter_performance = data.groupby("Letter")["Answered_Correctly"].mean()

import matplotlib.pyplot as plt

plt.figure()

letter_performance.sort_values(ascending=False).plot(kind="line")

plt.title("Recall Performance by Stimulus Letter")
plt.xlabel("Letter")
plt.ylabel("Average Performance (Accuracy)")
plt.ylim(0,1)
plt.xticks(range(len(letter_performance)), letter_performance.index)
plt.show()

letter_performance.plot(kind="line")

plt.title("Recall Performance by Stimulus Letter")
plt.xlabel("Letter")
plt.ylabel("Average Recall Accuracy")
plt.ylim(0,1)
plt.xticks(range(len(letter_performance)), letter_performance.index)
plt.show()

data["Answered_Correctly"].value_counts().plot(kind="bar")

plt.title("Distribution of Correct vs Incorrect Responses")
plt.xlabel("Response")
plt.ylabel("Count")

plt.show()
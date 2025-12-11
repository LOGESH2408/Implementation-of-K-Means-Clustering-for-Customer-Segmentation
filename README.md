# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess data: Import data, inspect it, and handle missing values if any.

2.Determine optimal clusters: Use the Elbow Method to identify the number of clusters by plotting WCSS against cluster numbers.

3.Fit the K-Means model: Apply K-Means with the chosen number of clusters to the selected features.

4.Assign cluster labels to each data point.

5.Plot data points in a scatter plot, color-coded by cluster assignments for interpretation

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: LOGESHWARAN S

RegisterNumber:25007255  
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
print(data.info())
print(data.isnull().sum())

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(data.iloc[:, 3:5])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.grid(True)
plt.show()


km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred

km = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = km.fit_predict(data.iloc[:, 3:5])
data["Cluster"] = y_pred




```

## Output:

<img width="708" height="554" alt="Screenshot 2025-12-11 091642" src="https://github.com/user-attachments/assets/4bbc438c-f214-426e-845d-0824a5b7e484" />

<img width="935" height="575" alt="Screenshot 2025-12-11 091701" src="https://github.com/user-attachments/assets/c81b01d8-3a12-452c-9197-ad8795652099" />

<img width="944" height="631" alt="Screenshot 2025-12-11 091725" src="https://github.com/user-attachments/assets/4650bef4-2141-4653-a56d-d99a84e40d77" />




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

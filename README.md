import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("C:\\Users\\User\\Desktop\\Новая папка (2)\\Library_Usage.csv")

# Выбор нужных столбцов
selected_columns = ["Total Checkouts", "Total Renewals", "Age Range"]
X = data[selected_columns]

# Преобразование столбца "Age Range" в числовой формат
le = LabelEncoder()
X.loc[:, 'Age Range'] = le.fit_transform(X['Age Range'])

# Преобразование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучение GMM
n_components = 3
gmm_model = GaussianMixture(n_components=n_components, random_state=42)
gmm_clusters = gmm_model.fit_predict(X_scaled)

# Обучение KMeans
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_clusters = kmeans_model.fit_predict(X_scaled)

# Получение предсказаний от каждого базового алгоритма
gmm_predictions = gmm_model.predict(X_scaled)
kmeans_predictions = kmeans_model.predict(X_scaled)

# Создание композиции с использованием голосования
ensemble_predictions = pd.DataFrame({'gmm': gmm_predictions, 'kmeans': kmeans_predictions})
ensemble_clusters = ensemble_predictions.mode(axis=1)[0]

# Добавление кластеров к данным
data['GMM_Cluster'] = gmm_clusters
data['KMeans_Cluster'] = kmeans_clusters
data['Ensemble_Cluster'] = ensemble_clusters

# Визуализация результатов
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_clusters, cmap='viridis', marker='o')
plt.title('GMM Clusters')

plt.subplot(132)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters, cmap='viridis', marker='o')
plt.title('KMeans Clusters')

plt.subplot(133)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=ensemble_clusters, cmap='viridis', marker='o')
plt.title('Ensemble Clusters')

plt.show()


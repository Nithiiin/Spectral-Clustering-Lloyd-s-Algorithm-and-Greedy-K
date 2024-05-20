
## Files

- `code.ipynb`: Jupyter notebook containing the code for clustering algorithms and data visualization.
- `ShapedData.csv`: Dataset with shaped data points.
- `clustering.csv`: Dataset for clustering analysis.

## Clustering Algorithms

### Lloyd's Algorithm (K-means)

The implementation of Lloyd's algorithm, also known as K-means clustering, is provided in the notebook. The algorithm iteratively updates the centroids and assigns data points to the closest centroid until convergence.

### Spectral Clustering

The notebook also includes spectral clustering using Gaussian similarity function and K-nearest neighborhood structure. Spectral clustering involves creating a weighted adjacency matrix and computing the eigenvectors of the Laplacian matrix.

## Usage

1. **Mount Google Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Import Libraries**:
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import NearestNeighbors
    ```

3. **Load Datasets**:
    ```python
    shapeddata_df = pd.read_csv("/content/drive/MyDrive/IE529_comp2/ShapedData.csv", header=None, names=['x','y'])
    clustering_df = pd.read_csv("/content/drive/MyDrive/IE529_comp2/clustering.csv", header=None, names=['x','y'])
    ```

4. **Lloyd's Algorithm Implementation**:
    ```python
    def lloydalgorithm(data_points, k, tolerance, maximum_iteration=10000):
        random_index = np.random.choice(data_points.shape[0], k, replace=False)
        k_centroids = data_points[random_index]
        for i in range(maximum_iteration):
            old_centroids = k_centroids
            diff = data_points - k_centroids.reshape(k_centroids.shape[0], 1, k_centroids.shape[1])
            dist = np.sqrt((diff**2).sum(axis=2))
            closest_pt = np.argmin(dist, axis=0)
            k_centroids = np.array([data_points[closest_pt==i].mean(axis=0) for i in range(k)])
            if np.all(np.abs(k_centroids - old_centroids)<tolerance):
                break
        best_distortion = 0
        for i, point_val in enumerate(k_centroids):
            cluster_points = data_points[closest_pt==i]
            best_distortion = best_distortion+np.sum((cluster_points-point_val)**2)
        return closest_pt, k_centroids,best_distortion
    ```

5. **Run Lloyd's Algorithm**:
    ```python
    labels, centroids, distortion = lloydalgorithm(np.array(clustering_df),4,1e-5)
    print("(i) a matrix of centroids : \n",centroids)
    print("\n(ii) a cluster index vector :\n", labels)
    print("\nBest Distortion value: ",distortion/len(np.array(clustering_df)))
    ```

6. **Visualize Lloyd's Algorithm**:
    ```python
    plt.figure(figsize=(10, 6))
    plt.scatter(np.array(clustering_df)[:, 0], np.array(clustering_df)[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=50, c='black', marker='+', label='Centroids')
    plt.title('Lloyd\'s Algorithm with k Clusters')
    plt.xlabel('Feature 1(x)')
    plt.ylabel('Feature 2(y)')
    plt.legend()
    plt.grid(True)
    plt.show()
    ```

7. **Spectral Clustering Implementation and Visualization**:
    The notebook contains several cells for implementing and visualizing spectral clustering using different similarity functions and neighborhood structures. 

## Results

The notebook prints and visualizes the centroids, cluster index vectors, and distortion values for the Lloyd's algorithm and spectral clustering methods.

## Dependencies

- pandas
- numpy
- matplotlib
- scikit-learn

## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Files

- `code.ipynb`: Jupyter notebook containing the code for clustering algorithms and data visualization.
- `ShapedData.csv`: Dataset with shaped data points.
- `clustering.csv`: Dataset for clustering analysis.

## Clustering Algorithms

### Lloyd's Algorithm (K-means)

The implementation of Lloyd's algorithm, also known as K-means clustering, is provided in the notebook. The algorithm iteratively updates the centroids and assigns data points to the closest centroid until convergence.

### Greedy K Algorithm

The Greedy K algorithm selects initial centroids greedily by choosing points that are farthest from the already chosen centroids, ensuring a diverse initial set of centroids.

### Spectral Clustering

The notebook also includes spectral clustering using Gaussian similarity function and K-nearest neighborhood structure. Spectral clustering involves creating a weighted adjacency matrix and computing the eigenvectors of the Laplacian matrix.




7. **Spectral Clustering Implementation and Visualization**:
    The notebook contains several cells for implementing and visualizing spectral clustering using different similarity functions and neighborhood structures. 

## Results

The notebook prints and visualizes the centroids, cluster index vectors, and distortion values for the Lloyd's algorithm and spectral clustering methods.


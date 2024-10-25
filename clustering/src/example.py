import numpy as np

def single_linkage_clustering(distance_matrix):
    # Number of initial clusters (one for each data point)
    n = distance_matrix.shape[0]
    
    # Initialize clusters
    clusters = [[i] for i in range(n)]
    
    # While there is more than one cluster
    while len(clusters) > 1:
        # Find the two closest clusters
        min_distance = np.inf
        cluster_a, cluster_b = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate the distance between clusters
                dist = single_linkage_distance(clusters[i], clusters[j], distance_matrix)
                if dist < min_distance:
                    min_distance = dist
                    cluster_a, cluster_b = i, j
        
        # Merge the closest clusters
        merged_cluster = clusters[cluster_a] + clusters[cluster_b]
        clusters.append(merged_cluster)
        
        # Remove the merged clusters
        clusters.pop(max(cluster_a, cluster_b))  # Remove the higher index first
        clusters.pop(min(cluster_a, cluster_b))
        
        # Update the distance matrix
        distance_matrix = update_distance_matrix(clusters, distance_matrix)

    return clusters[0]  # Return the final merged cluster

def single_linkage_distance(cluster_a, cluster_b, distance_matrix):
    # Single linkage distance: minimum distance between points in different clusters
    return min(distance_matrix[i, j] for i in cluster_a for j in cluster_b)

def update_distance_matrix(clusters, distance_matrix):
    n = len(clusters)
    new_distance_matrix = np.zeros((n, n))
    
    # Populate the new distance matrix
    for i in range(n):
        for j in range(i + 1, n):
            if i < n - 1 and j < n - 1:  # existing clusters
                new_distance_matrix[i, j] = distance_matrix[i, j]
            else:  # new cluster formed by merging
                new_distance_matrix[i, j] = single_linkage_distance(clusters[i], clusters[j], distance_matrix)

    # Symmetrize the new distance matrix
    new_distance_matrix = new_distance_matrix + new_distance_matrix.T
    return new_distance_matrix[:n-1, :n-1]  # Drop last row and column

# Example usage
if __name__ == "__main__":
    # Example distance matrix (symmetric)
    distance_matrix = np.array([
        [0, 1, 2, 3],
        [1, 0, 4, 5],
        [2, 4, 0, 6],
        [3, 5, 6, 0]
    ])

    final_clusters = single_linkage_clustering(distance_matrix)
    print("Final merged cluster:", final_clusters)

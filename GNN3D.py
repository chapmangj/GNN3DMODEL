import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go

class DrillHoleGNN(torch.nn.Module):
    """
    Rapid, Spatially-Aware Drill Hole Clustering and Automated 3D Modelling via GNNs
    """
    def __init__(self, num_features, hidden_channels=64, num_classes=None):
        super(DrillHoleGNN, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=1, dropout=0.2)
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1, dropout=0.2)
        self.num_classes = num_classes
        
        if num_classes:
            self.out = torch.nn.Linear(hidden_channels, num_classes)
        else:
            self.out = torch.nn.Linear(hidden_channels, hidden_channels // 2)
            
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return self.out(x) 
    
    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        embeddings = self.conv2(x, edge_index) 
        return embeddings

def read_and_process_data(filename="processed_drillhole_data clusters.csv"):
    """Read and preprocess drill hole data."""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found in current directory")
        return None, None
    
    print(f"Reading data from {filename}...")
    try:
        df = pd.read_csv(filename)
        if len(df.columns) == 1 and ';' in df.iloc[0, 0]:
            df = pd.read_csv(filename, sep=';')
        elif len(df.columns) == 1 and '\t' in df.iloc[0, 0]:
            df = pd.read_csv(filename, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None
    
    print(f"Data loaded successfully! Shape: {df.shape}")
    
    element_cols = ['Au', 'Ag', 'Al', 'As', 'Ba', 'Be', 'Bi', 'Ca', 'Cd', 'Ce', 'Co', 'Cr', 
                    'Cs', 'Cu', 'Fe', 'Ga', 'Ge', 'Hf', 'In', 'K', 'La', 'Li', 'Mg', 'Mn', 
                    'Mo', 'Na', 'Nb', 'Ni', 'P', 'Pb', 'Rb', 'Re', 'S', 'Sb', 'Sc', 'Se', 
                    'Sn', 'Sr', 'Ta', 'Te', 'Th', 'Ti', 'Tl', 'U', 'V', 'W', 'Y', 'Zn', 'Zr']
    
    feature_cols = [col for col in element_cols if col in df.columns]
    print(f"Identified feature columns: {feature_cols}")
    
    if not feature_cols:
        print("Error: No element/feature columns found in the data.")
        return None, None

    coord_cols = ['x', 'y', 'z']
    if not all(col in df.columns for col in coord_cols):
        print(f"Error: Missing one or more coordinate columns: {coord_cols}")
        return None, feature_cols

    for col in feature_cols + coord_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    missing_before = df[feature_cols].isnull().sum().sum()
    if missing_before > 0:
        print(f"Filling {missing_before} missing values in feature columns with column means...")
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())
    
    missing_coords = df[coord_cols].isnull().sum().sum()
    if missing_coords > 0:
        print(f"Error: {missing_coords} missing values found in coordinate columns.")
        return None, feature_cols

    return df, feature_cols

def create_compositional_similarity_graph(df, feature_cols, alpha=0.5, k_neighbors=15, sigma=50.0):
    """
    Create a graph based on compositional similarity and Gaussian spatial similarity.
    """
    print(f"Creating similarity graph (alpha={alpha}, k_neighbors={k_neighbors}, sigma={sigma})...")
    coord_cols = ['x', 'y', 'z']
    if not all(col in df.columns for col in coord_cols):
        print(f"Error: Missing coordinate columns {coord_cols}. Cannot create graph.")
        return None
    
    max_nodes = 10000
    if len(df) > max_nodes:
        print(f"Limiting to {max_nodes} nodes for performance")
        df_subset = df.iloc[:max_nodes].copy().reset_index(drop=True)
    else:
        df_subset = df.copy()

    X = df_subset[feature_cols].values
    if X.shape[0] == 0:
        print("Error: No data points after subsetting.")
        return None
    
    # Log10 transform with a small epsilon to avoid zeros
    epsilon = 1e-10  
    X_positive = np.maximum(X, epsilon)
    X_log = np.log10(X_positive)
    X_log = np.nan_to_num(X_log, nan=0.0, posinf=0.0, neginf=0.0)
    
    coords = df_subset[['x', 'y', 'z']].values
    
    print("Computing pairwise similarities...")
    try:
        comp_dist = pdist(X_log, metric='correlation')
        comp_similarity = 1 - squareform(comp_dist)
        comp_similarity = np.nan_to_num(comp_similarity, nan=0.0) 
    except Exception as e:
        print(f"Error calculating correlation similarity: {e}. Using Euclidean distance.")
        comp_dist_euc = pdist(X_log, metric='euclidean')
        median_dist_sq = np.median(comp_dist_euc)**2
        if median_dist_sq == 0: 
            median_dist_sq = 1.0
        comp_similarity = np.exp(-squareform(comp_dist_euc)**2 / (2 * median_dist_sq))

    np.fill_diagonal(comp_similarity, 0)

    spatial_distances = squareform(pdist(coords, metric='euclidean'))
    sigma_sq = sigma**2 + 1e-9 
    spatial_sim = np.exp(-spatial_distances**2 / (2 * sigma_sq))
    np.fill_diagonal(spatial_sim, 0)

    combined_similarity = alpha * comp_similarity + (1 - alpha) * spatial_sim
    
    print("Creating edges based on combined similarity...")
    edges = []
    num_nodes = combined_similarity.shape[0]
    if num_nodes <= 1:
        print("Error: Only one or zero nodes available. Cannot create edges.")
        return None
        
    for i in range(num_nodes):
        similarities = combined_similarity[i].copy()
        similarities[i] = -np.inf  # Do not consider self-loop
        actual_k = min(k_neighbors, num_nodes - 1)
        if actual_k <= 0: 
            continue
        most_similar_indices = np.argsort(similarities)[-actual_k:]
        for j in most_similar_indices:
            if 0 <= j < num_nodes and i != j:
                edges.append([i, j])
            
    if not edges:
        print("Warning: No edges created. Check similarity calculation or parameters.")
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x_tensor = torch.tensor(X_log, dtype=torch.float)
    
    data = Data(x=x_tensor, edge_index=edge_index)
    print(f"Graph created with {data.num_nodes} nodes and {data.num_edges} edges")
    return data

def train_gnn_model(data, num_epochs=100):
    """Train a GNN model on the drill hole graph."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if data is None:
        print("Error: No data provided to train_gnn_model.")
        return None
    data = data.to(device)
    
    if data.num_features is None or data.num_features == 0:
        print("Error: Data has no features. Cannot initialise model.")
        return None
         
    model = DrillHoleGNN(num_features=data.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    print("Training GNN model...")
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out, torch.zeros_like(out)) 
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    
    print("Training complete!")
    return model

def extract_embeddings(model, data):
    """Extract embeddings from the trained GNN model."""
    if model is None or data is None:
        print("Error: Model or data missing for embedding extraction.")
        return None
        
    device = next(model.parameters()).device
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(data.x, data.edge_index) 
    return embeddings.cpu().numpy()

def discover_clusters_from_gnn(embeddings_np, n_clusters=5, method='kmeans'):
    """
    Discover clusters from GNN embeddings using K-means.
    """
    print(f"\nDiscovering clusters using method: {method}")
    if embeddings_np is None or len(embeddings_np) == 0:
        print("Error: No embeddings provided for clustering.")
        return np.array([])
        
    n_samples = len(embeddings_np)
    if n_samples < n_clusters:
        print(f"Warning: Number of samples ({n_samples}) is less than n_clusters ({n_clusters}). Adjusting n_clusters to {n_samples}.")
        n_clusters = n_samples
    if n_clusters <= 0:
        print("Error: n_clusters must be positive. Setting to 1.")
        n_clusters = 1

    print(f"Using K-means with n_clusters = {n_clusters}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_np)

    unique_clusters, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique_clusters, counts))
    
    print(f"Found {len(unique_clusters)} clusters.")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} samples ({count/n_samples*100:.1f}%)")
    
    return clusters

def visualise_clusters_with_convex_hull(df, clusters, title="3D Drill Holes Clusters (Plotly)", ignore_ids=None):

    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull
    import numpy as np

    coord_cols = ['x', 'y', 'z']
    if not all(col in df.columns for col in coord_cols):
        print(f"Error: DataFrame must contain coordinate columns: {coord_cols}")
        return

    # Prepare the DataFrame for plotting using only the nodes corresponding to the clusters.
    df_plot = df.iloc[:len(clusters)].copy()
    # Convert clusters to string for discrete (categorical) colour mapping.
    df_plot['cluster'] = clusters.astype(str)
    
    # Ensure there's a 'sample_id' column; create one from the index if not present.
    if 'sample_id' not in df_plot.columns:
        df_plot['sample_id'] = df_plot.index

    # Use a specific discrete colour sequence.
    color_sequence = px.colors.qualitative.Plotly

    # Create the scatter plot with discrete colours.
    fig = px.scatter_3d(
        df_plot,
        x='x',
        y='y',
        z='z',
        color='cluster',
        hover_data=['sample_id'],
        title=title,
        color_discrete_sequence=color_sequence
    )

    # Remove legend items for scatter traces so that only the meshes appear in the legend.
    for trace in fig.data:
        if trace.type == 'scatter3d':
            trace.update(showlegend=False, legendgroup="")

    # Configure legend layout.
    fig.update_layout(
        legend=dict(
            title="Meshes",
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )

    # Build a mapping from cluster (as a string) to a colour from the same discrete colour sequence.
    unique_clusters = sorted(df_plot['cluster'].unique())
    cluster_color_map = {
        cl: color_sequence[i % len(color_sequence)]
        for i, cl in enumerate(unique_clusters)
    }

    if ignore_ids is None:
        ignore_ids = []

    # Compute and overlay the convex hull for each unique cluster.
    for clust in unique_clusters:
        # Filter points belonging to this cluster.
        cluster_points = df_plot[df_plot['cluster'] == clust][['x', 'y', 'z', 'sample_id']]
        # Exclude points with IDs in ignore_ids.
        if ignore_ids:
            cluster_points = cluster_points[~cluster_points['sample_id'].isin(ignore_ids)]
            
        pts = cluster_points[['x', 'y', 'z']].values
        if pts.shape[0] < 4:
            print(f"Cluster {clust} does not have enough points for a convex hull (requires at least 4) after ignoring specified IDs.")
            continue
        try:
            hull = ConvexHull(pts)
        except Exception as e:
            print(f"Error computing convex hull for cluster {clust}: {e}")
            continue
        
        simplices = hull.simplices
        
        # Use the same colour as assigned by the scatter plot.
        mesh_color = cluster_color_map[clust]
        
        mesh = go.Mesh3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            i=simplices[:, 0],
            j=simplices[:, 1],
            k=simplices[:, 2],
            opacity=0.3,
            color=mesh_color,
            name=f"Cluster {clust}",
            legendgroup=f"mesh_cluster_{clust}",
            showlegend=True
        )
        fig.add_trace(mesh)
    
    return fig

def main():
    # Define parameters
    input_file = 'processed_drillhole_data clusters.csv'
    n_clusters = 4
    alpha = 0.7
    k_neighbors = 25
    sigma = 300.0
    epochs = 100
    
    print("--- Starting GNN Analysis ---")
    print(f"Input file: {input_file}")
    print(f"n_clusters: {n_clusters}")
    print(f"Alpha (Composition vs Spatial): {alpha}")
    print(f"K-Neighbors: {k_neighbors}")
    print(f"Sigma (Spatial Similarity Bandwidth): {sigma}")
    print(f"GNN Epochs: {epochs}")
    print("-" * 30)

    df, feature_cols = read_and_process_data(input_file)
    if df is None or not feature_cols:
        print("Exiting: Data loading failed or no features identified.")
        return
        
    if df.empty:
         print("Exiting: DataFrame is empty after loading/processing.")
         return
    
    # --- Graph Construction ---
    data = create_compositional_similarity_graph(df, feature_cols, alpha=alpha, k_neighbors=k_neighbors, sigma=sigma)
    if data is None or data.num_nodes == 0:
        print("Exiting: Failed to create graph.")
        return
         
    # --- GNN Training ---
    model = train_gnn_model(data, num_epochs=epochs)
    if model is None:
        print("Exiting: Failed to train model.")
        return
        
    # --- Embedding Extraction ---
    print("\nExtracting GNN embeddings...")
    embeddings_np = extract_embeddings(model, data)
    if embeddings_np is None:
        print("Exiting: Failed to extract embeddings.")
        return
    print(f"Extracted embeddings shape: {embeddings_np.shape}")
        
    # --- Clustering ---
    gnn_clusters = discover_clusters_from_gnn(embeddings_np, n_clusters=n_clusters)
    
    # Prepare a DataFrame for visualisation (using only the nodes used in graph construction)
    df_clustered = df.iloc[:data.num_nodes].copy()
    
    # --- Initial Visualisation with no exclusions ---
    fig = visualise_clusters_with_convex_hull(df_clustered, clusters=gnn_clusters,
                                              title="3D Visualisation of Drill Holes Clusters")
    if fig is not None:
        fig.show()
    
    # --- Prompt for sample IDs to ignore in convex hull computation ---
    user_input = input("Enter sample IDs to ignore for mesh computation (comma separated) or press Enter to skip: ").strip()
    if user_input:
        try:
            ignore_ids = [int(x.strip()) for x in user_input.split(',') if x.strip() != ""]
        except ValueError:
            print("Invalid input. No samples will be ignored.")
            ignore_ids = []
        # Re-run visualisation to update the convex hulls
        fig = visualise_clusters_with_convex_hull(df_clustered, clusters=gnn_clusters,
                                                  title="3D Visualisation of Drill Holes Clusters (Filtered)",
                                                  ignore_ids=ignore_ids)
        if fig is not None:
            fig.show()
    
    # --- Option to download CSV with cluster labels appended ---
    download_csv = input("Would you like to download a CSV with cluster labels appended? (y/n): ").strip().lower()
    if download_csv.startswith('y'):
        # Append cluster labels to the DataFrame (for the nodes used in the graph)
        df_clustered['cluster'] = gnn_clusters
        output_filename = input("Enter output CSV filename (default: clustered_drillhole_data.csv): ").strip()
        if not output_filename:
            output_filename = "clustered_drillhole_data.csv"
        df_clustered.to_csv(output_filename, index=False)
        print(f"CSV with cluster labels saved as {output_filename}")
    
    print("\n--- GNN Analysis Complete ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n--- An unexpected error occurred ---")
        print(f"Error: {str(e)}")
        traceback.print_exc()

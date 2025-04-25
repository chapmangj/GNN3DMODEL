# GNN3DMODEL
Rapid, Spatially-Aware Drill Hole Clustering and Automated 3D Geological Modelling via GNNs

Let's build a first-pass 3D geological domain model directly from drill hole assays - in about 10 seconds.

While detailed geological modelling is a complex task, often relying on time-consuming manual logging interpretation or intricate geostatistical analysis, recent advancements allow for incredibly rapid initial domaining using quantitative data. Geologists often rely on clustering algorithms like K-means for initial pattern finding in drill hole assays. However, these methods ignore spatial context and the inherent anisotropy often present in geological deposits. While geostatistics explicitly models these features, it can be complex and time-consuming. What if we could generate spatially coherent, geologically plausible 3D domain models directly from assay data, adapting to the data's structure, and achieve automated modelling rapidly?
This post explores such an approach, leveraging Graph Neural Networks (GNNs) to learn representations from drill hole data by considering both geochemical similarity and spatial proximity simultaneously. By building connections based on neighbourhood similarity, this workflow implicitly adapts to underlying anisotropy without requiring predefined parameters. The result is a rapid pipeline, demonstrated here using Python, Pandas, PyTorch Geometric, and Plotly, that moves from raw assays to interactive, automatically generated 3D domain visualisations almost instantaneously.

The Core Idea: Representing Data as a Spatially-Aware Graph
This is where a data-driven geochemical approach using GNNs offers a powerful complement. Instead of relying solely on visual interpretation, it leverages quantitative assay data. It constructs a graph where each sample interval is a node. Edges connect nodes based on a blend of geochemical and spatial similarity, allowing the graph structure itself to reflect the underlying geology.
The process involves:
Data Preparation: Load desurvyed multi-element drill hole data (X, Y, Z, assays). Log-transform assays (handling zeros).
Choose the Number of Clusters (K): Based on geological knowledge and exploratory analysis of the embeddings
Similarity Calculation: Compute pairwise compositional similarity and spatial similarity for all data points.
Combined Similarity: Blend these using a weighted score
Edge Creation: Connect each node to its k most similar neighbours​. This step naturally encodes spatial structure and potential anisotropy into the graph's connectivity.

Learning from the Graph: The GAT Model
A Graph Attention Network (GAT) then learns from this structured data. The GAT generates an embedding vector for each node by aggregating information from its neighbours, weighted by attention scores. Because the graph encodes spatial relationships, the resulting embeddings are inherently spatially aware. A simple two-layer GAT trained unsupervisedly is sufficient to produce powerful embeddings using this method.

The Novelty: Speed, Spatial Awareness, and Automated Modelling
This GNN-based workflow stands apart from traditional methods:
Speed: The entire process, from data loading through GNN training, embedding extraction, clustering, and 3D visualisation, can often complete in tens of seconds for typical datasets, offering near real-time feedback for rapid iteration.
Objectivity: Relies on quantitative assay data, reducing the subjectivity inherent in visual logging.
Integrated Spatial Context: Embeddings inherently encode spatial relationships.
Adaptation to Anisotropy: The graph construction implicitly captures directional trends without needing explicit anisotropy models.
Automated 3D Visualisation: Quickly generate tangible 3D models (convex hulls) representing the clustered domains directly from the data and cluster results.

Tuning for Geological Insight: Hyperparameter Guidance
Fine-tuning this GNN approach involves adjusting key levers that control how the model perceives relationships in your data:
alpha (α, 0 to 1): The Geochemistry vs. Space Dial. Increasing α makes the model prioritize geochemical similarity, potentially creating clusters that group similar assays even if they are further apart spatially. Decreasing α emphasizes spatial proximity, leading to more spatially compact clusters, potentially merging geochemically distinct but adjacent units. Start around 0.5-0.7 and adjust based on whether results seem too fragmented spatially or too mixed geochemically.
sigma (σ, distance units): The Spatial Influence Range. This sets how far the model "looks" for spatial neighbours. A smaller σ (e.g., tens of metres) restricts influence locally, potentially separating structures within drill fences. A larger σ (e.g., hundreds of metres, matching drill spacing) allows connections across larger distances, promoting broader, more continuous domains that span multiple holes.
k-neighbours: Graph Connectivity. This controls how many neighbours each sample connects to based on the combined similarity. Increasing K (e.g., 30-50) creates a denser graph, potentially smoothing results and linking more distant samples. Decreasing kk (e.g., 10-20) makes the graph sparser, focusing on the strongest local similarities, which might lead to more detailed but potentially more fragmented clusters. 

def main():
    # Define parameters
    input_file = 'drillhole_data.csv'
    n_clusters = 4
    alpha = 0.7
    k_neighbors = 25
    sigma = 350.0
    epochs = 100p
    
From Embeddings to Interactive 3D Models: The Visualisation Step
This is where the speed and automation truly shine. Once the GNN produces embeddings and a clustering algorithm (like K-means) assigns cluster labels:
Extract Embeddings & Cluster: Obtain the vectors and assign cluster IDs.
Automated Visualisation with Convex Hulls: We use a dedicated function, visualise_clusters_with_convex_hull, leveraging Plotly for interactivity. This function automatically:

Creates a 3D plot of the drill hole samples, coloured by their assigned GNN cluster.
Includes hover information, displaying the sample_id (or index) when you mouse over a point, allowing quick identification.
Crucially, for each cluster, it computes the convex hull - the smallest convex shape enclosing all points in that cluster.

Interpret: Analyse the resulting interactive 3D plot. Do the coloured points and their corresponding hull shapes form geologically sensible domains? Do they align with known structures or logged units? The speed allows for rapid iteration - adjust hyperparameters, re-run, and see the updated automated 3D model in seconds.

Conclusion
This GNN-based workflow demonstrates the process of initial geological domaining. By combining the representational power of Graph Neural Networks with graph structures that encode both geochemistry and spatial proximity, it generates spatially coherent embeddings that implicitly adapt to data anisotropy, offering a powerful, objective complement to traditional logging. The ability to rapidly cluster these embeddings and instantly visualise the results as interactive, automatically generated 3D models with convex hulls provides geologists with an incredibly fast and intuitive tool for exploratory data analysis and rapid geological modelling directly from assay data. This tool has been built for demonstartion purposed bu if its utility can be demonstarted I will integrate it into 3DA (https://medium.com/@gavinjchapman/3da-rapid-drilling-data-visualisation-and-analysis-for-geologists-628af98



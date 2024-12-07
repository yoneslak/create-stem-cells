import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define parameters for synthetic data
num_samples = 1000  # Number of synthetic cells
gene_expression_range = (0, 100)  # Range for gene expression levels
stimulus_levels = ['low', 'medium', 'high']  # Possible stimulus levels
morphology_features = ['round', 'elongated', 'irregular']  # Morphological types

# Step 2: Generate synthetic data
# Generate random gene expression data
gene_expression_data = np.random.uniform(gene_expression_range[0], gene_expression_range[1], num_samples)

# Generate random stimulus levels
stimulus_data = np.random.choice(stimulus_levels, num_samples)

# Generate random morphology features
morphology_data = np.random.choice(morphology_features, num_samples)

# Step 3: Create a DataFrame
synthetic_data = pd.DataFrame({
    'Gene_Expression': gene_expression_data,
    'Stimulus_Level': stimulus_data,
    'Morphology': morphology_data
})

# Step 4: Analyze the synthetic data
def analyze_data(data):
    # Group by stimulus level and calculate mean gene expression
    mean_expression_by_stimulus = data.groupby('Stimulus_Level')['Gene_Expression'].mean()
    print("Mean Gene Expression by Stimulus Level:")
    print(mean_expression_by_stimulus)

    # Count of each morphology type
    morphology_counts = data['Morphology'].value_counts()
    print("\nCount of Each Morphology Type:")
    print(morphology_counts)

    # Summary statistics of gene expression
    summary_statistics = data['Gene_Expression'].describe()
    print("\nSummary Statistics of Gene Expression:")
    print(summary_statistics)

    return mean_expression_by_stimulus, morphology_counts, summary_statistics

# Initial analysis
mean_expression, morphology_counts, summary_statistics = analyze_data(synthetic_data)

# Step 5: PDE Simulation for Diffusion
# Parameters for the PDE simulation
D = 0.1       # Diffusion coefficient
L = 10.0      # Length of the domain
Nx = 100      # Number of spatial points
Nt = 1000     # Number of time steps
dx = L / (Nx - 1)  # Spatial step size
dt = 0.01     # Time step size

# Initial condition: a Gaussian pulse
x = np.linspace(0, L, Nx)
u = np.exp(-((x - L/2)**2) / 0.1)  # Initial concentration distribution

# Time-stepping loop for the PDE simulation
for n in range(Nt):
    u_new = u.copy()
    for i in range(1, Nx - 1):
        u_new[i] = u[i] + D * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u = u_new

# Step 6: Integrate PDE results into synthetic data
# Normalize the PDE output to match the gene expression range
u_normalized = (u - np.min(u)) / (np.max(u) - np.min(u)) * (gene_expression_range[1] - gene_expression_range[0])
gene_expression_data_with_pde = np.random.choice(u_normalized, num_samples)

# Update the synthetic data DataFrame with PDE-modulated gene expression
synthetic_data['Gene_Expression'] = gene_expression_data_with_pde

# Step 7: Re-analyze the synthetic data with updated gene expression
mean_expression_updated, morphology_counts_updated, summary_statistics_updated = analyze_data(synthetic_data)

# Step 8: Save the synthetic data to a CSV file
synthetic_data.to_csv('synthetic_gene_expression_data.csv', index=False)
print("\nSynthetic data saved to 'synthetic_gene_expression_data.csv'.")

# Step 9: Visualization functions
def plot_mean_expression(mean_expression):
    mean_expression.plot(kind='bar', color='skyblue', alpha=0.7)
    plt.title('Mean Gene Expression by Stimulus Level')
    plt.xlabel('Stimulus Level')
    plt.ylabel('Mean Gene Expression')
    plt.xticks(rotation=45)
    plt.show()

def plot_morphology_counts(morphology_counts):
    morphology_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightskyblue', 'lightgreen'])
    plt.title('Morphology Type Distribution')
    plt.ylabel('')  # Hide the y-label
    plt.show()

# Step 10: Visualize the results
plot_mean_expression(mean_expression_updated)
plot_morphology_counts(morphology_counts_updated)

# Optional: Plot the final gene expression distribution
plt.hist(synthetic_data['Gene_Expression'], bins=30, alpha=0.7, color='blue')
plt.title('Distribution of Gene Expression After PDE Integration')
plt.xlabel('Gene Expression Level')
plt.ylabel('Frequency')
plt.show()
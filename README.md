# Cancer-Gene Association Analysis

## Overview

This project analyzes genetic associations between cancer and other health conditions (such as depression or Alzheimer's disease). It automatically retrieves mutation data from cBioPortal, processes frequency information, and performs clustering analysis to identify meaningful patterns.

## Prerequisites

- Python 3.12.X
- Docker and Docker Compose (optional, for containerized execution)
- Internet connection (for cBioPortal API access)

## Installation

### Option 1: Python Environment with uv

1. **Install uv** (Python package manager):
   ```bash
   pip install uv
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/Aptroide/cancer_genes
   cd cancer_genes
   ```

3. **Initialize and install dependencies with uv**:
   ```bash
   # Create virtual environment and install all dependencies from pyproject.toml
   uv sync
   ```
   This command will:
   - Create a virtual environment in `.venv/` directory
   - Install all dependencies specified in `pyproject.toml`
   - Use the exact versions from `uv.lock` for reproducible builds

4. **Install Chrome for plotly visualizations**:
   ```bash
   # Install Chrome browser for static plot generation
   uv run plotly_get_chrome
   ```
   When prompted, type `y` to confirm the installation.

5. **Create required directories**:
   ```bash
   mkdir -p Data Results
   ```

6. **Create configuration file** (inside `/Data` directory):
   ```bash
   echo '{
       "file_path": "Data/your_condition_file.csv",
       "studies_path": null,
       "output_path": "Results",
       "num_cancer_studies": 10,
       "num_genes_per_study": null,
       "your_study_name": "Your_Condition",
       "classification": false
   }' > Data/config.json
   ```

7. **Prepare your input data file**:
   Create a CSV file with gene frequencies in `Data/your_condition_file.csv`:
   ```csv
   Gene,Frequency
   APP,0.45
   PSEN1,0.32
   PSEN2,0.28
   ```
   Format: Gene name in first column, frequency in second column

8. **Run the application**:
   ```bash
   # Run with uv (automatically uses the virtual environment)
   uv run main.py
   ```
   
   **Alternative**: If you prefer to suppress GPU warnings:
   - Linux and macOS:
   ```bash
   uv run main.py 2>/dev/null
   ```
   - Windows PowerShell:
   ```bash
   uv run main.py 2>$null
   ```

**Notes:**
- uv automatically manages the virtual environment, no need to activate/deactivate manually
- All dependencies are locked for reproducible installations
- Chrome is required for generating static plots with plotly/kaleido

### Option 2: Docker Deployment

Ensure Docker and Docker Compose are installed on your system. For installation instructions, visit the [Docker Desktop Documentation](https://docs.docker.com/desktop/).

1. Download the docker-compose.yml file:
   ```bash
   curl -O https://raw.githubusercontent.com/Aptroide/cancer_genes/main/docker-compose.yml
   ```

2. Create the required directories and files:
   ```bash
   mkdir -p ./Data ./Results/Figures ./Results/ClustersCSV
   ```

3. Create a basic configuration file:
   ```bash
   echo '{
       "file_path": "Data/your_condition_file.csv",
       "studies_path": null,
       "output_path": "Results",
       "num_cancer_studies": 10,
       "num_genes_per_study": null,
       "your_study_name": "Your_Condition",
       "classification": false
   }' > ./Data/config.json
   ```

4. Prepare your input data file:
   - Create a CSV file with gene frequencies in `Data/your_condition_file.csv`:

      ```bash
      Gene,Frequency
      APP,0.45
      PSEN1,0.32
      PSEN2,0.28
      ```

   - Format should be: Gene name in first column, frequency in second column

5. Build and start the container:
   ```bash
   docker-compose up
   ```

## Configuration

The `./Data` directory should contain your configuration file and input data files.

Configure the analysis parameters in the `Data/config.json` file:

```json
{
    "file_path": "Data/<Condition Frequencies File.csv>",
    "studies_path": "Data/<study ids.txt>",
    "output_path": "Results",
    "num_cancer_studies": 500,
    "num_genes_per_study": null,
    "your_study_name": "Condition_Name",
    "classification": false
}
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `file_path` | Path to CSV file containing condition-related genes (requires two columns: gene names and their frequencies) |
| `studies_path` | Path to text file with cBioPortal study IDs (one per line). Use `null` to fetch all available studies |
| `output_path` | Directory where result visualizations will be saved |
| `num_cancer_studies` | Maximum number of cancer studies to analyze (use lower values like 10 for testing) |
| `num_genes_per_study` | Maximum number of genes to analyze per study (use `null` to include all genes) |
| `your_study_name` | Name of your condition study (e.g., "Alzheimer", "Depression") |
| `classification` | Set to `true` to train ML and NN models for the best cluster files, `false` to disable this feature |

### Input File Requirements

1. **Condition Gene Data (`file_path`)**:
   - Must be a CSV file with two columns
   - Example format:
     ```
     Gene,Frequency
     APP,0.45
     PSEN1,0.32
     PSEN2,0.28
     ...
     ```

2. **Study IDs (`studies_path`)**:
   - Text file with one cBioPortal study ID per line
   - Example:
     ```
     blca_tcga
     brca_tcga
     coadread_tcga
     ...
     ```

## Results

### Output Files

Analysis results are saved in the `Results/ClustersCSV` directory:

| File | Description |
|------|-------------|
| `gene_comparison.csv` | Combined and filtered frequency data |
| `K-means_labels_<method>.csv` | K-means clustering results for different dimensionality reduction methods (UMAP, KPCA, t-SNE) |
| `DBSCAN_labels_<method>.csv` | DBSCAN clustering results for different methods |
| `times.csv` | Processing time statistics |

### Output Visualizations

Clustering visualizations are saved to the `Results/Figures` directory in PNG format:

| File | Description |
|------|-------------|
| `kmeans_<method>.png` | K-means clustering visualizations |
| `dbscan_<method>.png` | DBSCAN clustering visualizations |

**Note:** `<method>` can be: `kpca`, `tsne`, or `umap`.

## Troubleshooting

1. **API Connection Issues**: 
   - Ensure you have a stable internet connection
   - Verify that the cBioPortal API is accessible (https://www.cbioportal.org/api/v2/api-docs)

2. **Missing Results**:
   - Confirm that your input files follow the required format
   - Check that the specified output directories exist and are writable

3. **Docker Issues**:
   - Verify that the Docker service is running
   - Ensure required ports are available (port 8000 is used by default)

## References

- cBioPortal: https://www.cbioportal.org/

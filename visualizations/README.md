# Visualization

This folder contains Jupyter notebooks used to reproduce the alignment and survival analysis results presented in the manuscript (Figures 2, 4, and 5).  
Each notebook visualizes specific aspects of the THERAPI model, including embedding alignment, tissue-specific results, and clinical survival outcomes.

---

## Files

### `alignment_results_tcga.ipynb`
Generates alignment results between GDSC and unlabeled TCGA data using the model trained with `train_aligner.py`.  
This notebook saves:
- GDSC latent representations  
- TCGA attention weights  
- TCGA latent representations  
- TCGA weighted expression profiles  

---

### `alignment_results_external.ipynb`
Performs tissue-specific alignment visualization using the model trained with `train_aligner_tissue_specific.py`.  
For example, when aligning GDSC and External (breast) datasets, this notebook saves:
- GDSC latent representations  
- External attention weights  
- External latent representations  
- External weighted expression profiles  

---

### `main_figure2.ipynb`
Reproduces Figure 2 by visualizing the embedding spaces of GDSC and TCGA before and after alignment.  
To ensure consistent plotting, the TCGA sample order is fixed using `TCGA_ID_order.txt`.

---

### `main_figure4(a).ipynb`
Reproduces Figure 4a, showing the embedding space of the External dataset before and after alignment.

---

### `main_figure5(b).ipynb`
Reproduces Figure 5b, performing entropy-based grouping using the attention weights from the model trained with `train_aligner.py`.  
This notebook visualizes the Kaplan–Meier survival analysis results comparing groups with high and low entropy (tumor heterogeneity).

---

### `TCGA_ID_order.txt`
Specifies the TCGA sample order used for reproducible visualization across notebooks.

---

## Usage
Open the corresponding notebook in Jupyter and execute all cells.  
Ensure that model checkpoints (`ckpts/`) and datasets (`data/`) are properly placed as described in the main project README.

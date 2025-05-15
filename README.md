# THERAPI

THERAPI (Tumor Heterogeneity-aware Embedding for Response Adaptation and Patient Inference) is a deep learning framework that bridges the domain gap between preclinical and clinical data by modeling tumor heterogeneity and transferring gene-level drug-induced perturbation signatures to predict patient-specific drug responses.

## Model description

The full model architecture is provided below. THERAPI consists of two steps;

Step 1. Cell line-Patient Alignment: During the cell line and patient alignment step, patient samples are represented as linear combination of multiple cell lines through attention-based aggregation.

Step 2. Drug response prediction: The MLP predictor is first trained on cell line data using drug-induced perturbation (from CSG$^2$A) and rank-based gene expression embeddings (from Geneformer). Once trained, patient samples—transformed into weighted cell line representations from the alignment step—are passed through the predictor to infer drug response, without any further fine-tuning.

## Contact
If you have any questions or concerns, please send an email to [inyoung.sung@snu.ac.kr](inyoung.sung@snu.ac.kr).

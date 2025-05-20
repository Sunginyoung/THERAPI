# THERAPI

THERAPI (Tumor Heterogeneity-aware Embedding for Response Adaptation and Patient Inference) is a deep learning framework that bridges the domain gap between preclinical and clinical data by modeling tumor heterogeneity and transferring gene-level drug-induced perturbation signatures to predict patient-specific drug responses.

## Model description

The full model architecture is provided below. THERAPI consists of two steps;

Step 1. Cell line-Patient Alignment

Step 2. Drug response prediction

![model1](img/Overview.png)

## Setup
First, clone this repository and move to the directory.
```
git clone https://github.com/Sunginyoung/THERAPI.git
```

To install the appropriate environment for THERAPI, create a virtual environment and install the requirements befor running the code.
```
conda create -n [ENVIRONMENT NAME] python==3.9
conda activate [ENVIRONMENT NAME]
pip install -r requirements.txt
```

## Running THERAPI
#### 1. Download the dataset
```
cd THERAPI
mkdir data
cd data
```
Download the dataset used for the model from [Google Drive](https://drive.google.com/drive/folders/1rnYXYwwqwfqS-q50D68EnYjkUUfw2an2?usp=drive_link) to the `THERAPI/data` folder

#### 2. Training aligner
```
cd ../src
python train_aligner.py
```
Running the code above aligns the source domain data (cancer cell line data from GDSC) with the target domain data (patient tumor data from TCGA).
The trained alignment model is saved in the `ckpts` folder.

#### 3. Training drug response predictor
```
python train_predictor.py
```
Running the above code learns the drug response prediction model using the source domain data (cancer cell line data from GDSC). The models are trained in 10 folds.
The trained prediction models are saved in the `ckpts` folder.

#### 4. Predicting patient drug response
```
python test_TCGA.py
```
Running the above code predicts the drug response of the target domain data (patient tumor data from TCGA).
The predicted value is stored in the `output` folder


## Contact
If you have any questions or concerns, please send an email to [inyoung.sung@snu.ac.kr](inyoung.sung@snu.ac.kr).

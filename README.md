# SLOPE: Learning the Irreversible Progression Trajectory of Alzheimer’s Disease  

## Overview  
SLOPE (Semi-supervised Learning of Progressive Embeddings) is a novel machine learning framework designed to model the irreversible progression of Alzheimer's Disease (AD). Unlike traditional models that produce fluctuating risk predictions, SLOPE enforces a monotonic increase in predicted disease risk, ensuring biologically meaningful and consistent disease trajectories.  

## Key Features  
- **Monotonic Disease Progression**: Enforces directional loss to maintain increasing disease severity over time.  
- **Autoencoder-Based Latent Representations**: Captures disease trajectory from longitudinal imaging data.  
- **Pseudotime-Based Disease Modeling**: Uses embeddings to reconstruct AD progression across time.  
- **Comparison with Baselines**: Evaluates SLOPE against models without directional loss and traditional classifiers.  

## Dataset  
The model is trained on the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** amyloid PET imaging dataset, which consists of longitudinal data from 961 subjects across cognitive normal (CN), early mild cognitive impairment (EMCI), late MCI (LMCI), and AD stages.  

## Methodology  
### 1. Data Preprocessing  
- Standardized amyloid PET features from ADNI.  
- Adjusted for age, gender, and education to reduce biases.  

### 2. Model Components  
- **Autoencoder**: Compresses input features into meaningful latent representations.  
- **Triplet Loss**: Encourages similarity within the same class and separation between different classes.  
- **Directional Loss**: Enforces monotonicity in disease progression.  

### 3. Loss Function  
The total loss function combines multiple objectives:  
```math
L_{total} = \lambda_1 L_{rec} +  \lambda_2 L_{dir} 
```  
where:  
- `L_rec`: Reconstruction loss for autoencoder.  
- `L_dir`: Directional loss for monotonicity.  

## Experimental Results  
- **Violation Ratio & Violation Gap**: SLOPE significantly reduces instances of decreasing risk scores compared to baselines.  
- **Embedding Trajectories**: SLOPE ensures smoother, more biologically relevant disease trajectories.  
- **Classification Performance**: Outperforms traditional models in CN vs. AD classification with higher F1 scores and balanced accuracy.  

## Installation & Usage  
### Prerequisites  
- Python 3.8+  
- TensorFlow/PyTorch  
- NumPy, Pandas, SciPy  
- Scikit-learn, UMAP, Slingshot  

### Running the Model  
```bash
git clone https://github.com/your-repo/slope-ad.git
cd slope-ad
python train.py --dataset path/to/data
```  

## Results  
```latex
\begin{table}[H]
\centering
\setlength{\tabcolsep}{3pt} % Adjust column spacing
\scriptsize % Reduce font size
\begin{tabular}{|l|l|l|l|l|}
\hline
\textbf{Metric} & \textbf{Slope} & \textbf{Slope Without Direction Loss} & \textbf{LNE} & \textbf{Original Data} \\
\hline
 & \multicolumn{4}{c|}{\textbf{Logistic Regression}} \\
\hline
F1-Score & \textbf{0.841} & 0.800 & 0.814 & 0.771 \\
Balanced Accuracy & \textbf{0.858} & 0.825 & 0.836 & 0.803 \\
ROC-AUC & \textbf{0.858} & 0.825 & 0.836 & 0.803 \\
\hline
 & \multicolumn{4}{c|}{\textbf{Elastic Net}} \\
\hline
F1-Score & \textbf{0.841} & 0.814 & 0.822 & 0.776 \\
Balanced Accuracy & \textbf{0.858} & 0.836 & 0.840 & 0.805 \\
ROC-AUC & \textbf{0.858} & 0.836 & 0.840 & 0.805 \\
\hline
 & \multicolumn{4}{c|}{\textbf{MLP}} \\
\hline
F1-Score & \textbf{0.814} & 0.786 & 0.805 & 0.750 \\
Balanced Accuracy & \textbf{0.836} & 0.814 & 0.827 & 0.790 \\
ROC-AUC & \textbf{0.836} & 0.814 & 0.827 & 0.790 \\
\hline
\end{tabular}
\caption{Results for CN vs AD classification across preprocessing methods and classifiers.}
\label{tab:cn_vs_ad_transposed}
\end{table}
```  
## Conclusion  
SLOPE provides a robust approach to modeling Alzheimer’s Disease progression, ensuring monotonic risk predictions and improving classification accuracy. This method is applicable to any longitudinal disease modeling task requiring structured progression trajectories.

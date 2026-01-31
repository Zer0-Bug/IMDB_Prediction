<h1 align="center">IMDB Movie Rating Prediction</h1>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  </a>
  <a href="https://xgboost.readthedocs.io/">
    <img src="https://img.shields.io/badge/XGBoost-Regressor-green?style=for-the-badge" alt="XGBoost">
  </a>
  <a href="https://www.kaggle.com/">
    <img src="https://img.shields.io/badge/Dataset-Kaggle-00AFF0?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-darkred?style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  <b>An advanced regression pipeline for predicting movie ratings with high precision.</b><br><br>
  <i>Utilizing gradient boosting architectures and comprehensive feature engineering on extensive cinematic metadata.</i>
</p>
<br>
<p align="center">
  <a href="#technical-architecture">
    <img src="https://img.shields.io/badge/Architecture-222222?style=flat" />
  </a>
  <span> ° </span>
  <a href="#project-structure">
    <img src="https://img.shields.io/badge/Structure-222222?style=flat" />
  </a>
  <span> ° </span>
  <a href="#processing-pipeline">
    <img src="https://img.shields.io/badge/Pipeline-222222?style=flat" />
  </a>
  <span> ° </span>
  <a href="#detailed-module-specifications">
    <img src="https://img.shields.io/badge/Modules-222222?style=flat" />
  </a>
  <span> ° </span>
  <a href="#technical-specifications">
    <img src="https://img.shields.io/badge/Specs-222222?style=flat" />
  </a>
  <span> ° </span>
  <a href="#deployment--installation">
    <img src="https://img.shields.io/badge/Deploy-222222?style=flat" />
  </a>
</p>

---
<br>
<h2 align="center">Technical Architecture</h2>

The architectural design of this prediction system is optimized for high-dimensional cinematic data processing. The pipeline integrates several modular stages to ensure robust inference:

1.  **Data Acquisition & Rectification:** Ingestion of raw tabular data followed by rigorous cleansing to handle sparsity in fields such as budget, gross, and cast Facebook likes.
2.  **Multidimensional Feature Engineering:** Implementation of log transformations to normalize skewed financial distributions and label encoding for categorical cinematic attributes.
3.  **Ensemble Inference Layer:** A competitive model evaluation framework comparing traditional Regressors (Linear, Ridge, Lasso) against advanced ensemble methods including Random Forest and XGBoost.
4.  **Optimized Gradient Boosting:** Deployment of the XGBoost Regressor as the primary inference engine, leveraging second-order Taylor expansion for loss function optimization.

---
<br>
<h2 align="center">Project Structure</h2>

```
IMDB_Prediction/
├── LICENSE                                # MIT License
├── README.md                              # Project documentation
├── .gitattributes                         # Git configuration
│
├── Code and Dataset/                      # Core Machine Learning Sub-Project
│   ├── IMDB Movie Ratings Prediction.ipynb # Evolutionary development notebook
│   └── movie_metadata.csv                 # High-integrity dataset (5000+ records)
│
└── Documents/                             # Academic Reporting & Visualization
    ├── IEEE_Report.pdf                    # Formal technical analysis (IEEE Format)
    └── Poster.pdf                         # Visual data synthesis and results poster
```

---
<br>
<h2 align="center">Processing Pipeline</h2>

### 1. Data Integrity and Preprocessing
The system addresses data quality issues by isolating and rectifying missing values. Statistical imputation and removal strategies are applied to ensure a high signal-to-noise ratio before feature scaling.

### 2. Feature Transformation
Financial metrics such as 'budget' and 'gross' often exhibit massive variance. The pipeline applies log-base transformations to project these features into a more manageable Euclidean space for the regression models.

### 3. Model Benchmark Hierarchy
The system evaluates a spectrum of algorithms to establish a performance baseline:
-   **Linear Models:** establishing fundamental relationships via Linear and Ridge regression.
-   **Non-Linear Models:** Exploring local dependencies through K-Nearest Neighbors (KNN) and Decision Trees.
-   **Ensemble Architectures:** Utilizing Random Forests and XGBoost to minimize variance and bias.

### 4. Performance Optimization (XGBoost)
The final pipeline stage focuses on the XGBoost regressor, which achieves superior accuracy by iteratively refining predictions through tree-based residual reduction.

---
<br>
<h2 align="center">Detailed Module Specifications</h2>

### 1. Machine Learning Implementation (Code and Dataset/)
This module represents the computational core of the project. It handles the transition from raw CSV metadata to refined predictive insights:
-   **Notebook Logic:** The Jupyter environment documents the entire lifecycle from Exploratory Data Analysis (EDA) to hyperparameter tuning. It utilizes Seaborn and Matplotlib for deep visual inspection of feature correlations.
-   **Dataset Specifications:** The `movie_metadata.csv` contains attributes for over 5000 movies, spanning genres, directors, and financial performance metrics across multiple decades.
-   **Technical Optimization:** Implements advanced splitting techniques (80/20 Train-Test) and cross-validation to ensure model generalizability.

### 2. Technical Documentation & Synthesis (Documents/)
This sub-project focuses on the scholarly communication of the project results:
-   **IEEE Standard Reporting:** The `IEEE_Report.pdf` follows strict academic formatting to detail the methodology, mathematical foundations of the models used, and comprehensive result analysis.
-   **Visual Presentation Layer:** The `Poster.pdf` synthesizes complex data visualizations and model comparisons into an accessible format for technical presentations, highlighting the 97.86% accuracy achieved by the XGBoost model.

---
<br>
<h2 align="center">Technical Specifications</h2>

<table align="center">
  <tr>
    <th align="center">Metric</th>
    <th align="center">XGBoost (Best Performer)</th>
    <th align="center">Random Forest</th>
  </tr>
  <tr>
    <td align="center">RMSE (Test)</td>
    <td align="center">0.033</td>
    <td align="center">0.035</td>
  </tr>
  <tr>
    <td align="center">R2 Score (Test)</td>
    <td align="center">0.611</td>
    <td align="center">0.565</td>
  </tr>
  <tr>
    <td align="center">Model Accuracy</td>
    <td align="center">97.86%</td>
    <td align="center">97.74%</td>
  </tr>
</table>

<br>

<table align="center">
  <tr>
    <th align="center">Component</th>
    <th align="center">Technology Stack</th>
  </tr>
  <tr>
    <td align="center">Language</td>
    <td align="center">Python 3.8+</td>
  </tr>
  <tr>
    <td align="center">Core Libraries</td>
    <td align="center">Pandas, NumPy, Scikit-Learn</td>
  </tr>
  <tr>
    <td align="center">Ensemble Framework</td>
    <td align="center">XGBoost</td>
  </tr>
  <tr>
    <td align="center">Visualization</td>
    <td align="center">Matplotlib, Seaborn</td>
  </tr>
</table>

---
<br>
<h2 align="center">Deployment & Installation</h2>

### Repository Acquisition
To initialize a local copy of this project:
```bash
git clone https://github.com/Zer0-Bug/IMDB_Prediction.git
```

### Environment Configuration
Dependencies are managed via standard Python package managers. It is recommended to use a virtual environment:
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Execution
To reproduce the analysis and results:
1. Navigate to the `Code and Dataset` directory.
2. Launch Jupyter Notebook or JupyterLab.
3. Execute the cells in `IMDB Movie Ratings Prediction.ipynb` sequentially.

---
<br>
<h2 align="center">Contribution</h2>

Contributions are always appreciated. Open-source projects grow through collaboration, and any improvement—whether a bug fix, new feature, documentation update, or suggestion—is valuable.

To contribute, please follow the steps below:

1. Fork the repository.
2. Create a new branch for your change:  
   `git checkout -b feature/your-feature-name`
3. Commit your changes with a clear and descriptive message:  
   `git commit -m "Add: brief description of the change"`
4. Push your branch to your fork:  
   `git push origin feature/your-feature-name`
5. Open a Pull Request describing the changes made.
<br>
All contributions are reviewed before being merged. Please ensure that your changes follow the existing code style and include relevant documentation or tests where applicable.

---

<br>
<p align="center">
  <a href="mailto:777eerol.exe@gmail.com">
    <img src="https://cdn.simpleicons.org/gmail/D14836" width="40" alt="Email">
  </a>
  <span> × </span>
  <a href="https://www.linkedin.com/in/eerolexe/">
    <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
         width="40"
         alt="LinkedIn">
  </a>
</p>

---

<p align="center" style="margin-top:10px; letter-spacing:4px;">
  ∞
</p>

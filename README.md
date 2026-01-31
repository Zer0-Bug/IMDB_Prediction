<h1 align="center">IMDB Movie Ratings Prediction</h1>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  </a>
  <a href="https://xgboost.ai/">
    <img src="https://img.shields.io/badge/XGBoost-darkgreen?style=for-the-badge" alt="XGBoost">
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-darkred?style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  <b>An end-to-end Machine Learning pipeline for predicting movie ratings with high precision.</b><br><br>
  <i>Leveraging XGBoost, Random Forest, and advanced feature engineering to analyze cinematic success.</i>
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
  <a href="#model-performance-results">
    <img src="https://img.shields.io/badge/Results-222222?style=flat" />
  </a>
  <span> ° </span>
  <a href="#installation--usage">
    <img src="https://img.shields.io/badge/Deploy-222222?style=flat" />
  </a>
</p>

---
<br>
<h2 align="center">Technical Architecture</h2>

The project is structured as a robust regression pipeline designed to predict IMDB scores based on historical cinematic data. The workflow follows a systematic approach to ensure data integrity and model generalization:

1.  **Exploratory Data Analysis (EDA):** Statistical analysis of features such as genre, budget, and cast influence.
2.  **Data Preprocessing:** Sophisticated handling of missing values and categorical encoding using log transformations for skewed distributions.
3.  **Feature Engineering:** Extraction of meaningful insights from cast and crew data to enhance predictive power.
4.  **Model Selection & Evaluation:** Comparative analysis of multiple regressors (Linear, KNN, Decision Trees, Random Forest, XGBoost) to optimize for RMSE and R² metrics.

---
<br>
<h2 align="center">Project Structure</h2>

```
IMDB_Prediction/
├── LICENSE                                   # MIT License
├── README.md                                 # Project documentation
├── .gitattributes                            # Git configuration for attributes
│
├── Code and Dataset/                         # Core machine learning development
│   ├── IMDB Movie Ratings Prediction.ipynb   # Comprehensive Jupyter Notebook (EDA + Modeling)
│   └── movie_metadata.csv                    # Raw dataset containing 5000+ movie records
│
└── Documents/                                # Scientific and presentational materials
    ├── IEEE_Report.pdf                       # Technical research paper in IEEE format
    └── Poster.pdf                            # Visual summary and presentation poster
```

---
<br>
<h2 align="center">Processing Pipeline</h2>

### 1. Data Cleaning
The system identifies and handles missing values within the `movie_metadata.csv`. Features with excessive null values are pruned, while others are imputed based on statistical medians or modes.

### 2. Feature Transformation
To handle the high variance in movie budgets and gross earnings, log transformations are applied. This normalizes the distribution, allowing models like Linear Regression to perform more effectively.

### 3. Categorical Encoding
Categorical variables such as `genre` and `director_name` are transformed into numerical representations using encoding techniques, ensuring the models can ingest non-numeric cinematic data.

```python
# Conceptual transformation logic
df['gross_log'] = np.log1p(df['gross'])
df['budget_log'] = np.log1p(df['budget'])
```

### 4. Model Training & Testing
The dataset is split into training and testing sets (typically 75/25 or 80/20) to evaluate the model's ability to generalize to unseen movie data.

### 5. Performance Monitoring
During training, the system monitors Mean Squared Error (MSE) and R² Score. The XGBoost regressor is iteratively tuned to achieve the lowest possible error rates.

---
<br>
<h2 align="center">Detailed Module Specifications</h2>

### 1. Code and Dataset (Core Implementation)
This directory contains the primary intellectual property of the project.
- **IMDB Movie Ratings Prediction.ipynb:** This notebook is the heart of the project. It includes the full data science lifecycle: from importing `pandas` and `numpy` to visualizing data distributions with `seaborn`. It implements five distinct machine learning algorithms, providing a comparative framework for cinematic success prediction.
- **movie_metadata.csv:** A comprehensive dataset sourced from Kaggle, featuring 28 attributes for over 5000 movies. Key features include director names, lead actors, genres, and social media metrics (Facebook likes).

### 2. Documents (Scientific reporting)
This section provides the formal academic context for the project.
- **IEEE_Report.pdf:** A high-level technical document detailing the methodology, mathematical foundations of the algorithms used (e.g., the loss functions in XGBoost), and a deep dive into the results. It follows standard IEEE publication guidelines.
- **Poster.pdf:** A condensed, visual representation of the project designed for academic conferences or project showcases. It highlights the key findings, such as the superiority of ensemble methods over linear models.

---
<br>
<h2 align="center">Technical Specifications</h2>

<table align="center">
  <tr>
    <th align="center">Component</th>
    <th align="center">Details</th>
  </tr>
  <tr>
    <td align="center">Programming Language</td>
    <td align="center">Python 3.8+</td>
  </tr>
  <tr>
    <td align="center">Primary Libraries</td>
    <td align="center">Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn</td>
  </tr>
  <tr>
    <td align="center">Dataset Volume</td>
    <td align="center">5043 Records, 28 Columns</td>
  </tr>
  <tr>
    <td align="center">Preprocessing Techniques</td>
    <td align="center">Label Encoding, Log Transformation, Missing Value Imputation</td>
  </tr>
  <tr>
    <td align="center">Algorithm Suite</td>
    <td align="center">Linear, Lasso, Ridge Regression, KNN, Decision Tree, Random Forest, XGBoost</td>
  </tr>
  <tr>
    <td align="center">Development Environment</td>
    <td align="center">Jupyter Notebook / Anaconda</td>
  </tr>
</table>

---
<br>
<h2 align="center">Model Performance Results</h2>

<table align="center">
  <tr>
    <th align="center">Model</th>
    <th align="center">RMSE (Train)</th>
    <th align="center">RMSE (Test)</th>
    <th align="center">R² (Train)</th>
    <th align="center">R² (Test)</th>
    <th align="center">Accuracy</th>
  </tr>
  <tr>
    <td align="center">Linear Regression</td>
    <td align="center">0.119</td>
    <td align="center">0.120</td>
    <td align="center">0.411</td>
    <td align="center">0.387</td>
    <td align="center">95.43%</td>
  </tr>
  <tr>
    <td align="center">Decision Tree</td>
    <td align="center">0.029</td>
    <td align="center">0.049</td>
    <td align="center">0.716</td>
    <td align="center">0.148</td>
    <td align="center">97.0%</td>
  </tr>
  <tr>
    <td align="center">Random Forest</td>
    <td align="center">0.014</td>
    <td align="center">0.035</td>
    <td align="center">0.925</td>
    <td align="center">0.565</td>
    <td align="center">97.74%</td>
  </tr>
  <tr>
    <td align="center">KNN</td>
    <td align="center">0.040</td>
    <td align="center">0.049</td>
    <td align="center">0.443</td>
    <td align="center">0.159</td>
    <td align="center">96.74%</td>
  </tr>
  <tr>
    <td align="center">Lasso Regression</td>
    <td align="center">0.044</td>
    <td align="center">0.043</td>
    <td align="center">0.351</td>
    <td align="center">0.346</td>
    <td align="center">97.15%</td>
  </tr>
  <tr>
    <td align="center">Ridge Regression</td>
    <td align="center">0.044</td>
    <td align="center">0.043</td>
    <td align="center">0.351</td>
    <td align="center">0.346</td>
    <td align="center">97.15%</td>
  </tr>
  <tr>
    <td align="center"><b>XGBoost</b></td>
    <td align="center"><b>0.004</b></td>
    <td align="center"><b>0.033</b></td>
    <td align="center"><b>0.991</b></td>
    <td align="center"><b>0.611</b></td>
    <td align="center"><b>97.86%</b></td>
  </tr>
</table>

---
<br>
<h2 align="center">Deployment & Installation</h2>

### Repository Acquisition
To initialize a local instance of this repository, execute the following commands in your terminal:

```bash
git clone https://github.com/Zer0-Bug/IMDB_Prediction.git
```

```bash
cd IMDB_Prediction
```

### Environment Configuration
The project dependencies are managed via `pip`. It is highly recommended to utilize a isolated virtual environment to prevent dependency conflicts:
```bash
# Optional: Create and activate virtual environment
python -m venv venv

source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

# Install core dependencies
```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn jupyterlab
```

### Running the Analysis
The primary analytical engine is contained within the Jupyter Notebook. To reproduce the results:

1. Launch JupyterLab:
```bash
jupyter lab
```
2. Navigate to the `Code and Dataset/` folder via the sidebar.
3. Open `IMDB Movie Ratings Prediction.ipynb`.
4. Execute all cells (`Run > Run All Cells`) to observe the EDA and model benchmarking.

---
<br>
<h2 align="center">Future Improvements</h2>

- **Incorporating NLP Techniques**: Analyzing movie reviews to enhance prediction accuracy.
- **Using Deep Learning**: Implementing neural networks to capture complex relationships in the data.
- **Expanding Feature Set**: Adding social media metrics, box-office earnings, and critic scores.

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
<h2 align="center">References</h2>

1. **Sharda, R., & Delen, D. (2006)** –  
   [Predicting box-office success of motion pictures with neural networks](https://doi.org/10.1016/j.eswa.2005.07.018).  
   *Expert Systems with Applications, 31*(3), 481–490.

2. **Choudhury, M., & Gaonkar, S. (2018)** –  
   Predicting movie success using machine learning.  
   *Journal of Data Science, 16*(2), 95–110.

3. **Breiman, L. (2001)** –  
   [Random forests](https://doi.org/10.1023/A:1010933404324).  
   *Machine Learning, 45*(1), 5–32.

4. **Goodfellow, I., Bengio, Y., & Courville, A. (2016)** –  
   [Deep Learning](https://www.deeplearningbook.org).  
   MIT Press.

5. **Chen, T., & Guestrin, C. (2016)** –  
   [XGBoost: A scalable tree boosting system](https://arxiv.org/abs/1603.02754).  
   In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).

6. **Saurav, S. (2023)** –  
   [IMDB Score Prediction for Movies](https://www.kaggle.com/code/saurav9786/imdb-score-prediction-for-movies).  
   *Kaggle*.

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

# Analysis and Prediction of Customer Non-Payment Risk

A machine learning project to predict customer payment default risk for Compagnie Internationale de Leasing (CIL), implementing the CRISP-DM methodology for data science.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Business Objectives](#business-objectives)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models and Results](#models-and-results)
- [Files Description](#files-description)
- [Technologies Used](#technologies-used)
- [Key Findings](#key-findings)

## 🎯 Project Overview

This project aims to help the International Leasing Company (CIL) identify customers likely to default on payments using historical customer and contract data. Accurate prediction enables the company to:
- Reduce financial losses from payment defaults
- Improve leasing approval decisions
- Optimize collection strategies
- Enhance revenue security and profitability

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology, covering all phases from business understanding to deployment.

## 💼 Business Objectives

### Primary Goals
- **Reduce default rate**: Proactively identify high-risk customers to minimize payment defaults and associated losses
- **Improve profitability**: Enhance revenue security through better risk management
- **Support decision-making**: Provide data-driven insights for leasing approval, monitoring, and collection processes

### Analytical Objective
Build a classification model to estimate the probability of payment default over a 12-month horizon, supporting:
- Leasing approval decisions
- Customer monitoring
- Collection strategy optimization

## 📊 Dataset

The project uses leasing data from CIL Tunisia with the following characteristics:

- **Source**: `dataset_leasing.xlsx`
- **Target Variable**: `default_flag` (1 = default, 0 = no default)
- **Key Features**:
  - Customer demographics and credit information
  - Income and debt ratio
  - Credit score
  - Payment delays
  - Type of financed asset
  - Contract details

## 📁 Project Structure

```
Analysis-and-Prediction-of-Customer-Non-Payment-Risk/
├── projet_cil.ipynb                    # Main Jupyter notebook with complete analysis
├── dataset_leasing.xlsx                # Original dataset
├── clean_data_phase_1.csv             # Cleaned dataset after initial processing
├── data_phase_2_prepared.csv          # Fully prepared dataset ready for modeling
├── best_model_rf.pkl                  # Saved Random Forest model
└── README.md                          # Project documentation
```

## 🔧 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab

### Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost imbalanced-learn statsmodels scipy
pip install openpyxl  # For reading Excel files
```

Or install all dependencies at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn statsmodels scipy openpyxl
```

## 🚀 Usage

### Running the Analysis

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yousra0/Analysis-and-Prediction-of-Customer-Non-Payment-Risk.git
   cd Analysis-and-Prediction-of-Customer-Non-Payment-Risk
   ```

2. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook projet_cil.ipynb
   ```
   
   Or upload to Google Colab and run interactively.

3. **Follow the notebook sections**:
   - Business Understanding
   - Data Understanding & Exploration
   - Data Preparation
   - Modeling
   - Evaluation
   - Deployment

### Using the Pre-trained Model

```python
import pickle
import pandas as pd

# Load the saved model
with open('best_model_rf.pkl', 'rb') as file:
    model = pickle.load(file)

# Load prepared data
data = pd.read_csv('data_phase_2_prepared.csv')

# Make predictions
predictions = model.predict(data)
probabilities = model.predict_proba(data)
```

## 🔄 Methodology

This project follows the **CRISP-DM** methodology with six phases:

### 1️⃣ Business Understanding
- Define business and analytical objectives
- Understand the leasing industry context
- Identify key stakeholders and requirements

### 2️⃣ Data Understanding
- Initial data collection and exploration
- Data quality assessment
- Identify missing values and outliers
- Exploratory Data Analysis (EDA)
- Statistical analysis and visualizations
- Correlation analysis and VIF (Variance Inflation Factor)

### 3️⃣ Data Preparation
- Data cleaning and preprocessing
- Missing value imputation
- Outlier treatment (Winsorization)
- Feature encoding (One-Hot, Label Encoding)
- Feature engineering and creation
- Multicollinearity handling
- Feature scaling and normalization
- Class imbalance handling using SMOTE

### 4️⃣ Modeling
- Model selection and training:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Gradient Boosting
- Hyperparameter tuning using GridSearchCV
- Cross-validation

### 5️⃣ Evaluation
- Model performance comparison
- Metrics analysis:
  - ROC-AUC Score
  - Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
- Business validation
- Feature importance analysis

### 6️⃣ Deployment
- Model serialization (pickle)
- Documentation and recommendations
- Integration guidelines for production use

## 📈 Models and Results

### Model Performance Comparison

| Model | ROC-AUC | F1-Score | Key Characteristics |
|-------|---------|----------|---------------------|
| **XGBoost** | ~0.97 | ~0.97 | **Best performer**, selected for deployment |
| **Random Forest** | High | High | Strong alternative, robust performance |
| **Gradient Boosting** | Good | Good | Solid performance |
| **Logistic Regression** | Baseline | Baseline | Interpretable baseline model |

### Key Performance Insights

- **XGBoost** dominates with near-perfect scores (ROC-AUC ≈ 0.97)
- **Random Forest** provides excellent alternative with strong interpretability
- Both ensemble methods significantly outperform logistic regression
- Models successfully identify high-risk customers with high precision

### Important Features

Key predictors of payment default:
- Payment delays (>0.5 months indicates high risk)
- Debt ratio (>100% indicates elevated risk)
- Credit score
- Income level
- Asset type financed
- Customer profile characteristics

## 📄 Files Description

| File | Description |
|------|-------------|
| `projet_cil.ipynb` | Main Jupyter notebook containing complete analysis pipeline |
| `dataset_leasing.xlsx` | Original raw dataset from CIL |
| `clean_data_phase_1.csv` | Dataset after initial cleaning and quality checks |
| `data_phase_2_prepared.csv` | Final prepared dataset with feature engineering and preprocessing |
| `best_model_rf.pkl` | Serialized Random Forest model ready for deployment |
| `README.md` | Project documentation (this file) |

## 🛠 Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models and tools
- **imbalanced-learn**: SMOTE for class imbalance
- **XGBoost**: Gradient boosting implementation

### Data Analysis & Visualization
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualizations
- **statsmodels**: Statistical tests and VIF analysis
- **scipy**: Statistical functions

### Model Development
- **GridSearchCV**: Hyperparameter optimization
- **Pipeline**: Model workflow automation
- **Cross-validation**: Model validation

## 🔑 Key Findings

1. **Payment History is Critical**: Customers with payment delays >0.5 months have significantly higher default probability

2. **Debt Burden Matters**: Debt ratio >100% is a strong indicator of financial stress and default risk

3. **Model Performance**: XGBoost and Random Forest achieve excellent discrimination between defaulters and non-defaulters

4. **Feature Engineering Impact**: Created features (ratios, interactions) improve model performance significantly

5. **Class Imbalance**: SMOTE effectively addresses the imbalanced nature of default data

6. **Actionable Insights**: The model provides interpretable results that align with business understanding and can guide operational decisions

## 📝 Recommendations

1. **Deployment**: Integrate XGBoost model into CIL's customer evaluation system
2. **Monitoring**: Track payment delays and debt ratios as early warning indicators
3. **Risk Stratification**: Use probability scores to segment customers into risk categories
4. **Regular Updates**: Retrain model periodically with new data to maintain performance
5. **A/B Testing**: Validate model impact on business metrics through controlled experiments

## 👥 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This project was developed for educational and analytical purposes to demonstrate machine learning applications in financial risk management.

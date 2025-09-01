# 🔴 GlucoTrack – Advanced Track

## Project Overview

GlucoTrack is a comprehensive machine learning project focused on developing accurate diabetes risk prediction models using the CDC Diabetes Health Indicators dataset. The project aims to create clinically interpretable models that can assist healthcare professionals in early diabetes detection and risk stratification.

**Primary Goals:**
- Build robust diabetes risk prediction models with >90% sensitivity for clinical deployment
- Identify the most predictive health indicators for diabetes risk assessment
- Develop interpretable models suitable for healthcare decision support
- Ensure model fairness across demographic subgroups
- Create actionable insights for diabetes prevention strategies

**Dataset:** CDC Diabetes Health Indicators (UCI ML Repository ID: 891) containing 253,680 samples with 21 health-related features from the 2014 BRFSS survey.

---

## Week 1 Deliverables

### ✅ Completed Tasks

**1. Comprehensive Exploratory Data Analysis (EDA)**
- Complete dataset profiling and quality assessment
- Statistical analysis of all 21 features plus target variable
- Missing value analysis and data integrity validation
- Distribution analysis for numerical and categorical variables
- Correlation analysis and multicollinearity assessment
- Target variable analysis and class imbalance evaluation

**2. Data Quality Assessment**
- Identified and validated 253,680 total samples with 22 columns
- Confirmed zero missing values across all variables
- Detected 24,206 duplicate rows requiring preprocessing attention
- Validated all binary variables contain only 0/1 values
- Confirmed appropriate data types for all features

**3. Comprehensive EDA Report**
- Detailed answers to 15 specific research questions
- Data-driven insights with statistical evidence
- Clinical interpretation of findings
- Preprocessing recommendations
- Modeling strategy guidance

**4. Technical Documentation**
- Well-structured Jupyter notebook with 30 executable cells
- Professional visualizations and statistical summaries
- Comprehensive markdown documentation
- Repository organization and documentation

---

## Exploratory Data Analysis (EDA)

### 📊 High-Level Overview

Our EDA process followed a systematic approach covering six major areas:
1. **Setup and Data Loading** - Dataset acquisition and initial inspection
2. **Data Integrity Assessment** - Missing values, duplicates, and format validation
3. **Univariate Analysis** - Individual feature distributions and quality
4. **Bivariate Analysis** - Feature relationships and correlations
5. **Target Variable Analysis** - Class distribution and imbalance assessment
6. **Modeling Preparation** - Preprocessing recommendations and next steps

### 🔍 Key Findings

**Dataset Quality:**
- **Excellent completeness:** Zero missing values across 253,680 samples
- **Data integrity issue:** 24,206 duplicate rows (9.5%) requiring removal
- **Proper encoding:** All variables appropriately formatted for analysis
- **Clinical relevance:** All 21 features medically meaningful for diabetes prediction

**Target Variable Insights:**
- **Class distribution:** 86.07% no diabetes, 13.93% diabetes/prediabetes
- **Significant imbalance:** 6.2:1 ratio requiring specialized handling techniques
- **Sample sufficiency:** 35,346 positive cases adequate for robust modeling

**Feature Characteristics:**
- **4 Numerical variables:** BMI, MentHlth, PhysHlth, Age with distinct distribution patterns
- **18 Categorical variables:** Binary health indicators properly encoded as 0/1
- **Low multicollinearity:** Only one strong correlation (GenHlth ↔ PhysHlth: 0.524)
- **Feature independence:** Most variables provide unique predictive information

**Health Pattern Discoveries:**
- **BMI distribution:** Population skews toward overweight/obese (mean ~28-30)
- **Health day patterns:** Mental and physical health metrics show survey response clustering
- **Comorbidity clustering:** Diabetes strongly associated with hypertension, high cholesterol, heart disease
- **Socioeconomic gradients:** Education and income inversely correlated with diabetes risk
- **Age progression:** Clear age-related increase in diabetes prevalence

**Notable Outliers and Patterns:**
- **BMI outliers:** Some extreme values >60 representing severely obese individuals (clinically meaningful)
- **Health days clustering:** Peaks at 0, 15, and 30 days indicating survey response patterns
- **Binary variable validation:** All categorical features properly encoded with no invalid entries
- **Healthcare access patterns:** Financial barriers to care strongly associated with diabetes risk

**Data Quality Issues Addressed:**
- **Duplicate detection:** Systematic identification of 24,206 duplicate records
- **Range validation:** No unrealistic values detected in any numerical features
- **Encoding verification:** All binary variables contain only valid 0/1 values
- **Data readiness score:** 3/4 (excellent foundation for modeling)

---

## Repository Structure

```
/advanced/submissions/team-members/yan-cotta/
│
├── README.md                           # Project documentation and overview (this file)
├── REPORT.md                           # Comprehensive EDA and feature engineering report
├── week_1_eda.ipynb                    # Week 1: Complete exploratory data analysis
└── week_2_feature_engineering.ipynb    # Week 2: Feature engineering and deep learning prep
```

### 📁 File Descriptions

**`README.md`**
- Project overview and goals
- Week 1 deliverables summary
- High-level EDA findings
- Repository structure documentation
- Setup and execution instructions
- Next steps and roadmap

**`REPORT.md`**
- Detailed answers to 15 specific research questions about the dataset
- Comprehensive analysis of data quality, target distribution, and feature relationships
- Statistical evidence and clinical interpretations
- Preprocessing recommendations and modeling strategy
- Executive summary suitable for project stakeholders

**`week_1_eda.ipynb`**
- Interactive Jupyter notebook with 30 cells (6 sections)
- Complete data loading, cleaning, and analysis pipeline
- Professional visualizations including correlation heatmaps, distribution plots, and missing value analysis
- Statistical summaries and data validation frameworks
- Executable code with detailed markdown documentation
- Final data validation and modeling preparation insights

**`week_2_feature_engineering.ipynb`**
- Complete feature engineering pipeline with 26 executable cells
- Implementation of WHO BMI categorization and integer encoding for neural networks
- StandardScaler application and stratified data splitting
- PyTorch DataLoader creation and validation
- Comprehensive markdown documentation with senior-level analysis
- Production-ready preprocessing pipeline for deep learning implementation

---

## How to Run

### 📋 Prerequisites

**Required Python Packages:**
```bash
pip install pandas numpy matplotlib seaborn ucimlrepo jupyter
```

**Environment Setup:**
- Python 3.8+ recommended
- Jupyter Notebook or VS Code with Jupyter extension
- Minimum 4GB RAM for dataset processing
- Internet connection for initial dataset download

### 🚀 Execution Instructions

**1. Launch Jupyter Environment:**
```bash
# Option 1: Jupyter Notebook
jupyter notebook week_1_eda.ipynb

# Option 2: Jupyter Lab
jupyter lab week_1_eda.ipynb

# Option 3: VS Code
code week_1_eda.ipynb
```

**2. Package Installation (if needed):**
The notebook will automatically install required packages, or run:
```bash
pip install ucimlrepo pandas numpy matplotlib seaborn
```

**3. Execute Notebook:**
- Run all cells sequentially using "Run All" or execute cell-by-cell
- First execution will download the dataset (~42MB)
- Complete execution time: ~5-10 minutes depending on system
- All cells should execute without errors

**4. View Results:**
- Visualizations will render inline
- Statistical summaries displayed in output cells
- Final data validation and summary in last cells

### 🔧 Troubleshooting

**Common Issues:**
- **Import errors:** Ensure all packages installed with correct versions
- **Memory issues:** Close other applications if dataset loading fails
- **Network errors:** Check internet connection for dataset download
- **Kernel issues:** Restart kernel if variables become undefined

---

## Next Steps

### 📅 Week 2: Feature Engineering and Preprocessing - ✅ COMPLETE
**Completed Deliverables:**
- **Robust Preprocessing Pipeline:** Successfully removed 24,206 duplicate rows and optimized data types for 15.2% memory reduction
- **Strategic Feature Engineering:** Implemented WHO BMI categorization based on team health expert feedback, creating clinically meaningful categories
- **Neural Network Optimization:** Applied integer encoding to 3 high-cardinality categorical features for embedding layer compatibility
- **Perfect Feature Scaling:** Achieved optimal normalization (mean≈0, std≈1) for numerical features using StandardScaler
- **Stratified Data Splitting:** Maintained critical class balance (85.4%/14.6%) across 70/15/15 train/validation/test splits
- **Production-Ready DataLoaders:** Created optimized PyTorch DataLoaders with batch size 64 for efficient neural network training

**Key Accomplishments:** Completed a comprehensive preprocessing pipeline including duplicate removal, WHO BMI categorization based on team feedback, feature scaling, and stratified data splitting. Final dataset: 229,474 samples × 22 features, ready for neural network implementation.

### 📅 Week 3: Model Development and Training
**Planned Deliverables:**
- **Baseline Models:** Logistic Regression, Decision Trees for interpretable benchmarks
- **Advanced Algorithms:** Random Forest, XGBoost, Neural Networks
- **Hyperparameter Optimization:** Grid search and cross-validation
- **Ensemble Methods:** Model stacking and voting classifiers

### 📅 Week 4: Model Evaluation and Validation
**Planned Deliverables:**
- **Healthcare-Focused Metrics:** Sensitivity, specificity, NPV, clinical utility
- **Model Interpretability:** SHAP values, feature importance analysis
- **Bias Assessment:** Fairness evaluation across demographic subgroups
- **Clinical Validation:** Performance testing for real-world deployment

### 📅 Week 5: Deployment and Documentation
**Planned Deliverables:**
- **Production Model:** Optimized and validated diabetes risk classifier
- **Clinical Decision Support:** Risk calculator and intervention recommendations
- **Deployment Framework:** Model packaging and monitoring setup
- **Final Documentation:** Comprehensive project report and technical specifications

### 🎯 Success Metrics
- **Model Performance:** >90% Sensitivity, >80% Specificity, F1-score >0.85
- **Clinical Utility:** NPV >95% for diabetes risk exclusion
- **Interpretability:** Clear identification of top 5-7 risk factors
- **Fairness:** Consistent performance across all demographic groups
- **Deployment Readiness:** Production-ready model with monitoring framework

---

## 📞 Contact Information

**Project Lead:** Yan Cotta  
**Track:** Advanced  
**Dataset:** CDC Diabetes Health Indicators (UCI ML Repository ID: 891)  
**Analysis Date:** August 8, 2025

For questions about this analysis or to contribute to the project, please refer to the project repository documentation or contact the development team.

---

*This README represents the current state of Week 1 deliverables. The document will be updated as the project progresses through subsequent phases.*

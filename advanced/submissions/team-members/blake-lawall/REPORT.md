# 🔴 GlucoTrack – Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A:  1. Missing values: No
2. Duplicates: Yes, 24206 rows (9.54%)
3. Incorrectly formatted: Yes

Q: Are all data types appropriate (e.g., numeric, categorical)?  
A:   By Storage Type (how they're currently stored):
   Numerical (float64/int64): 22 features
   Categorical (object/category): 0 features

📋 By Actual Content (what they really represent):
   Binary features (0/1): 15 features
   - Diabetes_binary, HighBP, HighChol, CholCheck, Smoker
   - Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies
   - HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex

   Multi-class categorical: 3 features
   - GenHlth (5 categories: 1-5 scale)
   - Education (6 categories: 1-6 scale)
   - Income (8 categories: 1-8 scale)

   Continuous numerical: 4 features
   - BMI (continuous values)
   - MentHlth (0-30 days)
   - PhysHlth (0-30 days)
   - Age (18-80 years)

🔑 Key Insight:
   All 22 features are currently stored as float64, but only 4 are truly continuous numerical.
   The other 18 should be treated as categorical (15 binary + 3 multi-class) for proper ML preprocessing.

📈 This means you have:
   - 4 numerical features that need scaling
   - 18 categorical features that need encoding (15 binary + 3 multi-class)


Q: Did you detect any constant, near-constant, or irrelevant features?  
A:   Yes, the following features are constant or near-constant:
   - Diabetes_binary: 0.0000 unique ratio (2 unique values)
   - HighBP: 0.0000 unique ratio (2 unique values)
   - HighChol: 0.0000 unique ratio (2 unique values)
   - CholCheck: 0.0000 unique ratio (2 unique values)
   - BMI: 0.0003 unique ratio (84 unique values)
   - Smoker: 0.0000 unique ratio (2 unique values)
   - Stroke: 0.0000 unique ratio (2 unique values)
   - HeartDiseaseorAttack: 0.0000 unique ratio (2 unique values)
   - PhysActivity: 0.0000 unique ratio (2 unique values)
   - Fruits: 0.0000 unique ratio (2 unique values)
   - Veggies: 0.0000 unique ratio (2 unique values)
   - HvyAlcoholConsump: 0.0000 unique ratio (2 unique values)
   - AnyHealthcare: 0.0000 unique ratio (2 unique values)
   - NoDocbcCost: 0.0000 unique ratio (2 unique values)
   - GenHlth: 0.0000 unique ratio (5 unique values)
   - MentHlth: 0.0001 unique ratio (31 unique values)
   - PhysHlth: 0.0001 unique ratio (31 unique values)
   - DiffWalk: 0.0000 unique ratio (2 unique values)
   - Sex: 0.0000 unique ratio (2 unique values)
   - Age: 0.0001 unique ratio (13 unique values)
   - Education: 0.0000 unique ratio (6 unique values)
   - Income: 0.0000 unique ratio (8 unique values)

✅ SUMMARY:
   - Data quality: Issues found
   - Data types: Need conversion
   - Feature relevance: Some near-constant features

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of `Diabetes_binary`?  
A:  Distribution:
   Diabetes_binary=0.0: 218334 samples (86.1%)
   Diabetes_binary=1.0: 35346 samples (13.9%)

Q: Is there a class imbalance? If so, how significant is it?  
A:  Class imbalance ratio: 6.18:1
   ⚠️  Significant class imbalance detected!
   The majority class is 6.177049736886777 times larger than the minority class.


Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A:   Recommendations:
   - Use F1-score, precision, recall instead of accuracy
   - Consider class weights in models
   - Use SMOTE or other resampling techniques
   - Stratified sampling for train/test splits

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  Analysis of numerical features:

Skewed features (|skewness| > 1):
   - CholCheck: skewness = -4.88
   - BMI: skewness = 2.12
   - Stroke: skewness = 4.66
   - HeartDiseaseorAttack: skewness = 2.78
   - PhysActivity: skewness = -1.20
   - Veggies: skewness = -1.59
   - HvyAlcoholConsump: skewness = 3.85
   - AnyHealthcare: skewness = -4.18
   - NoDocbcCost: skewness = 3.00
   - MentHlth: skewness = 2.72
   - PhysHlth: skewness = 2.21
   - DiffWalk: skewness = 1.77

Features with outliers (> 1.5 * IQR):
   - CholCheck: 9470 outliers (3.7%)
   - BMI: 9847 outliers (3.9%)
   - Stroke: 10292 outliers (4.1%)
   - HeartDiseaseorAttack: 23893 outliers (9.4%)
   - PhysActivity: 61760 outliers (24.3%)
   - Veggies: 47839 outliers (18.9%)
   - HvyAlcoholConsump: 14256 outliers (5.6%)
   - AnyHealthcare: 12417 outliers (4.9%)
   - NoDocbcCost: 21354 outliers (8.4%)
   - GenHlth: 12081 outliers (4.8%)
   - MentHlth: 36208 outliers (14.3%)
   - PhysHlth: 40949 outliers (16.1%)
   - DiffWalk: 42675 outliers (16.8%)

Q: Did any features contain unrealistic or problematic values?  
A:  Value range analysis:
   HighBP: range [0.0, 1.0]
   HighChol: range [0.0, 1.0]
   CholCheck: range [0.0, 1.0]
   BMI: range [12.0, 98.0]
   Smoker: range [0.0, 1.0]
   Stroke: range [0.0, 1.0]
   HeartDiseaseorAttack: range [0.0, 1.0]
   PhysActivity: range [0.0, 1.0]
   Fruits: range [0.0, 1.0]
   Veggies: range [0.0, 1.0]
   HvyAlcoholConsump: range [0.0, 1.0]
   AnyHealthcare: range [0.0, 1.0]
   NoDocbcCost: range [0.0, 1.0]
   GenHlth: range [1.0, 5.0]
   MentHlth: range [0.0, 30.0]
   PhysHlth: range [0.0, 30.0]
   DiffWalk: range [0.0, 1.0]
   Sex: range [0.0, 1.0]
   Age: range [1.0, 13.0]
   Education: range [1.0, 6.0]
   Income: range [1.0, 8.0]

Q: What transformation methods (if any) might improve these feature distributions?  
A:  Recommended transformations:
   For skewed features:
     - CholCheck: Consider square root transformation
     - BMI: Consider log transformation or Box-Cox
     - Stroke: Consider log transformation or Box-Cox
     - HeartDiseaseorAttack: Consider log transformation or Box-Cox
     - PhysActivity: Consider square root transformation
     - Veggies: Consider square root transformation
     - HvyAlcoholConsump: Consider log transformation or Box-Cox
     - AnyHealthcare: Consider square root transformation
     - NoDocbcCost: Consider log transformation or Box-Cox
     - MentHlth: Consider log transformation or Box-Cox
     - PhysHlth: Consider log transformation or Box-Cox
     - DiffWalk: Consider log transformation or Box-Cox
   For features with outliers:
     - CholCheck: Consider capping outliers or robust scaling
     - BMI: Consider capping outliers or robust scaling
     - Stroke: Consider capping outliers or robust scaling
     - HeartDiseaseorAttack: Consider robust scaling or outlier removal
     - PhysActivity: Consider robust scaling or outlier removal
     - Veggies: Consider robust scaling or outlier removal
     - HvyAlcoholConsump: Consider robust scaling or outlier removal
     - AnyHealthcare: Consider capping outliers or robust scaling
     - NoDocbcCost: Consider robust scaling or outlier removal
     - GenHlth: Consider capping outliers or robust scaling
     - MentHlth: Consider robust scaling or outlier removal
     - PhysHlth: Consider robust scaling or outlier removal
     - DiffWalk: Consider robust scaling or outlier removal

---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A:  Categorical features with clear patterns:
   - GenHlth: 35.4% range (2.5% to 37.9% diabetes)
   - HeartDiseaseorAttack: 21.0% range (12.0% to 33.0% diabetes)
   - DiffWalk: 20.2% range (10.5% to 30.7% diabetes)
   - Stroke: 18.6% range (13.2% to 31.8% diabetes)
   - HighBP: 18.4% range (6.0% to 24.4% diabetes)
   - HighChol: 14.0% range (8.0% to 22.0% diabetes)
   - CholCheck: 11.8% range (2.5% to 14.4% diabetes)

Q: Are there any strong pairwise relationships or multicollinearity between features?  
A:  Correlation analysis:
   Strong correlations (|r| > 0.5):
     - GenHlth ↔ PhysHlth: r = 0.524

Q: What trends or correlations stood out during your analysis?  
A:  Key insights:
   Strongest categorical patterns:
     - GenHlth: 35.4% difference in diabetes rates
     - HeartDiseaseorAttack: 21.0% difference in diabetes rates
     - DiffWalk: 20.2% difference in diabetes rates

---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  Key takeaways:
   1. ⚠️  Data quality issues detected (missing values or duplicates)
   2. ⚠️  Significant class imbalance (6.2:1 ratio)
   3. 📊 Mixed feature types: 22 numerical, 0 categorical
   4. 📈 13 numerical features contain outliers
   5. 🔗 1 strong feature correlations detected

Q: Which features will you scale, encode, or exclude in preprocessing?  
A:  Preprocessing recommendations:
   Scaling needed:
     - HighBP: Standard scaling
     - HighChol: Standard scaling
     - CholCheck: Robust scaling (due to skewness/outliers)
     - BMI: Robust scaling (due to skewness/outliers)
     - Smoker: Standard scaling
     - Stroke: Robust scaling (due to skewness/outliers)
     - HeartDiseaseorAttack: Robust scaling (due to skewness/outliers)
     - PhysActivity: Robust scaling (due to skewness/outliers)
     - Fruits: Standard scaling
     - Veggies: Robust scaling (due to skewness/outliers)
     - HvyAlcoholConsump: Robust scaling (due to skewness/outliers)
     - AnyHealthcare: Robust scaling (due to skewness/outliers)
     - NoDocbcCost: Robust scaling (due to skewness/outliers)
     - GenHlth: Standard scaling
     - MentHlth: Robust scaling (due to skewness/outliers)
     - PhysHlth: Robust scaling (due to skewness/outliers)
     - DiffWalk: Robust scaling (due to skewness/outliers)
     - Sex: Standard scaling
     - Age: Standard scaling
     - Education: Standard scaling
     - Income: Standard scaling
   Encoding needed:
     - Sex: Binary encoding
     - HighChol: Binary encoding
     - NoDocbcCost: Binary encoding
     - CholCheck: Binary encoding
     - HvyAlcoholConsump: Binary encoding
     - Veggies: Binary encoding
     - PhysActivity: Binary encoding
     - Smoker: Binary encoding
     - Fruits: Binary encoding
     - AnyHealthcare: Binary encoding
     - DiffWalk: Binary encoding
     - Stroke: Binary encoding
     - HeartDiseaseorAttack: Binary encoding
     - HighBP: Binary encoding
     - GenHlth: One-hot encoding
   Exclusion candidates:
     - Diabetes_binary: Near-constant feature (2 unique values)
     - HighBP: Near-constant feature (2 unique values)
     - HighChol: Near-constant feature (2 unique values)
     - CholCheck: Near-constant feature (2 unique values)
     - BMI: Near-constant feature (84 unique values)
     - Smoker: Near-constant feature (2 unique values)
     - Stroke: Near-constant feature (2 unique values)
     - HeartDiseaseorAttack: Near-constant feature (2 unique values)
     - PhysActivity: Near-constant feature (2 unique values)
     - Fruits: Near-constant feature (2 unique values)
     - Veggies: Near-constant feature (2 unique values)
     - HvyAlcoholConsump: Near-constant feature (2 unique values)
     - AnyHealthcare: Near-constant feature (2 unique values)
     - NoDocbcCost: Near-constant feature (2 unique values)
     - GenHlth: Near-constant feature (5 unique values)
     - MentHlth: Near-constant feature (31 unique values)
     - PhysHlth: Near-constant feature (31 unique values)
     - DiffWalk: Near-constant feature (2 unique values)
     - Sex: Near-constant feature (2 unique values)
     - Age: Near-constant feature (13 unique values)
     - Education: Near-constant feature (6 unique values)
     - Income: Near-constant feature (8 unique values)

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A:  Expected cleaned dataset:
   Shape: (253680, 4)
   Rows: 253,680 samples
   Columns: 4 features (after encoding)
   Data types: All numerical (float64)
   Missing values: 0
   Duplicates: 0

---
---

## ✅ Week 2: Feature Engineering & Deep Learning Prep

---

### 🏷️ 1. Categorical Feature Encoding

Q: Which categorical features in the dataset have more than two unique values?  
A:  

Q: Apply integer-encoding to these high-cardinality features. Why is this strategy suitable for a subsequent neural network with an embedding layer?  
A:  

Q: Display the first 5 rows of the transformed data to show the new integer labels.  
A:  

---

### ⚖️ 2. Numerical Feature Scaling

Q: Which numerical features did your EDA from Week 1 suggest would benefit from scaling?  
A:  

Q: Apply a scaling technique to these features. Justify your choice of `StandardScaler` vs. `MinMaxScaler` or another method.  
A:  

Q: Show the summary statistics of the scaled data to confirm the transformation was successful.  
A:  

---

### ✂️ 3. Stratified Data Splitting

Q: Split the data into training, validation, and testing sets (e.g., 70/15/15). What function and parameters did you use?  
A:  

Q: Why is it critical to use stratification for this specific dataset?  
A:  

Q: Verify the stratification by showing the class distribution of `Diabetes_binary` in each of the three resulting sets.  
A:  

---

### 📦 4. Deep Learning Dataset Preparation

Q: Convert your three data splits into PyTorch `DataLoader` or TensorFlow `tf.data.Dataset` objects. What batch size did you choose and why?  
A:  

Q: To confirm they are set up correctly, retrieve one batch from your training loader. What is the shape of the features (X) and labels (y) in this batch?  
A:  

Q: Explain the role of the `shuffle` parameter in your training loader. Why is this setting important for the training set but not for the validation or testing sets?  
A:
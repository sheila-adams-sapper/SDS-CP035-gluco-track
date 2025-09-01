# 🟢 GlucoTrack – Beginner Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?  
A: No missing values were found in the dataset; all 253,680 instances were complete. A total of 24,206 rows (9.5% of the data) were flagged as duplicates and will be further assessed for removal to avoid bias from repeated observations. No formatting issues were present, but all features were stored as float64. Several of these variables should be represented as integers or categorical values. 

Q: Are all data types appropriate (e.g., numeric, categorical)?  
A:  Not all data types in the raw dataset were appropriately represented. While some measures such as BMI, mental health, and physical health are naturally numeric and correctly stored as floats, other features were misclassified. Binary categorical variables were stored as continuous floats, and ordinal variables lost their semantic meaning when represented in this way. To address this, explicit datatype corrections were required, informed by domain knowledge, to properly classify features as categorical, ordinal, or numeric.

Feature Type	Columns
Binary categorical	Outcome, Sex, HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk
Ordinal / Categorical	GenHlth, Age, Education, Income
Numeric	BMI, MentHlth, PhysHlth

Q: Did you detect any constant, near-constant, or irrelevant features?  
A: No constant features were detected, as all variables had at least two unique values.  At this stage, no features appeared irrelevant, and all were retained for further assessment in training.

---

### 🎯 2. Target Variable Assessment

Q: What is the distribution of `Diabetes_binary`?  
A:  After removing 24,602 duplicate instances, the target variable Diabetes_binary remained imbalanced, with 84.7% of observations labeled “0” (no diabetes) and 15.3% labeled “1” (diabetes positive). This pronounced skew toward the negative class underscores the importance of applying class-balance strategies such as weighting or resampling during model development to avoid bias toward the majority class.


Q: Is there a class imbalance? If so, how significant is it?  
A: Yes, the dataset shows a significant class imbalance. After removing 24,602 duplicate instances, 84.7% of observations belong to the negative class (no diabetes), while only 15.3% belong to the positive class (diabetes). This imbalance is substantial and requires mitigation during model training to prevent the classifier from being biased toward the majority class. 

Q: How might this imbalance influence your choice of evaluation metrics or model strategy?  
A:  The pronounced class imbalance means that accuracy alone would be a misleading evaluation metric, as a model could achieve high accuracy by predicting the majority class only. Instead, metrics that account for the minority class, such as precision, recall, F1-score, ROC-AUC, and PR-AUC, are more appropriate. From a modeling perspective, strategies such as class weighting, resampling (oversampling the minority or undersampling the majority), or using algorithms robust to imbalance (e.g., tree-based ensembles with balanced class weights) will be considered to ensure the model does not ignore the minority class of interest.

---

### 📊 3. Feature Distribution & Quality

Q: Which numerical features are skewed or contain outliers?  
A:  Using a Z-score threshold of 3, we identified 15,328 potential outlier rows across the numerical features BMI, MentHlth, and PhysHlth. This confirms what was visible in the histograms: MentHlth and PhysHlth are highly skewed, with a large number of values far from the mean. BMI also contributed to the outlier count, though less dramatically.

For GenHlth, although it is an ordinal categorical variable (1–5), extreme values at the higher “poor health” end still appear far from the mean relative to the standard deviation, which is consistent with the skewed distribution observed.

Overall, the Z-score analysis highlights that outliers are most pronounced in the health-related features (MentHlth and PhysHlth). These findings underscore the importance of considering strategies such as feature transformation, outlier handling, or robust modeling approaches to mitigate their impact.

Q: Did any features contain unrealistic or problematic values?  
A:  While the dataset contained no missing values or incorrectly formatted entries, several features—particularly BMI, MentHlth, and PhysHlth—show a substantial number of statistically unusual values. These outliers, though still within the defined ranges, are atypical relative to the overall distributions and may impact model performance. No variables were found to contain values outside their expected ranges or categories.

Q: What transformation methods (if any) might improve these feature distributions?  
A:  For MentHlth and PhysHlth, which are heavily right-skewed and have a large number of zeros, the Log transformation with log(x+1) or Square Root transformation could be good starting points.
For BMI, which is moderately right-skewed, any of the power transformations (Log, Square Root, Box-Cox, or Yeo-Johnson) or Quantile transformation could be considered.
The best transformation method often depends on the specific distribution of the feature and requires experimentation to see which one yields the most desirable distribution and improves model performance.
---

### 📈 4. Feature Relationships & Patterns

Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?  
A:  To examine whether categorical features show patterns in relation to Diabetes_binary, cross-tabulations were generated for GenHlth (General Health), PhysActivity (Physical Activity), and Smoker (Smoking).

GenHlth: A strong pattern is evident—individuals reporting worse general health have a much higher prevalence of diabetes compared to those reporting excellent or very good health.

PhysActivity: Those reporting no physical activity show a markedly higher diabetes rate (21.14%) than those who are physically active (11.61%).

Smoker: Smokers have a slightly higher diabetes prevalence (16.29%) compared to non-smokers (12.06%).

In summary, all three features demonstrate visible patterns, with general health and physical activity showing the strongest associations with diabetes, and smoking showing a weaker but still notable relationship.

Q: Are there any strong pairwise relationships or multicollinearity between features?  
A: Pairwise relationships and potential multicollinearity were assessed using a correlation matrix. Several moderate correlations were observed, such as GenHlth with PhysHlth (≈0.42), DiffWalk (≈0.32), and HighBP (≈0.30). PhysHlth was also moderately correlated with MentHlth (≈0.28) and DiffWalk (≈0.32). As expected, Income correlated with Education (≈0.45), and HighBP correlated with HighChol (≈0.30).

While these relationships are intuitive, none exceeded the common thresholds (0.7–0.8) that typically signal multicollinearity concerns. In summary, the dataset contains several moderate, meaningful associations, but no evidence of severe multicollinearity that would compromise models such as logistic regression or tree-based methods. 

Q: What trends or correlations stood out during your analysis?  
A: The analysis revealed several important trends and correlations. First, the target variable Diabetes_binary shows a strong class imbalance, with 84.7% of individuals not having diabetes and 15.3% having diabetes, a factor that must be addressed in model training. Among numerical features, MentHlth and PhysHlth are highly skewed with many outliers, while BMI also shows right skewness and some extreme values.

Clear relationships emerged between general health and diabetes: individuals reporting poorer overall health had a much higher prevalence of diabetes. Physical activity was linked with lower diabetes rates, while smoking showed a weaker but still notable association with higher rates. Correlation analysis revealed expected moderate relationships, such as between GenHlth and PhysHlth, and between Income and Education, but no severe multicollinearity.

Finally, features such as GenHlth, BMI, HighBP, and HighChol showed stronger positive correlations with Diabetes_binary, suggesting they may play an important role in predicting diabetes status. These insights emphasize the need to address imbalance, skewness, and outliers during preprocessing, while focusing on the most predictive health and demographic factors. 
---

### 🧰 5. EDA Summary & Preprocessing Plan

Q: What are your 3–5 biggest takeaways from EDA?  
A:  Severe Class Imbalance
The target Diabetes_binary is heavily skewed (84.7% no diabetes vs. 15.3% diabetes), making class imbalance a central challenge that will affect model choice and evaluation metrics.

Skewness and Outliers in Health Features
MentHlth and PhysHlth are highly skewed with many extreme values, while BMI also shows skewness and outliers. These distributions may distort models unless handled through transformation, robust methods, or outlier strategies.

Strong Link Between General Health and Diabetes
Poorer self-reported general health is strongly associated with higher diabetes prevalence, making GenHlth a particularly important predictor.

Lifestyle Factors Show Patterns
Physical activity is linked with substantially lower diabetes rates, while smoking is associated with somewhat higher rates. These lifestyle features provide meaningful predictive signal.

Moderate but Manageable Correlations
Expected relationships appear (e.g., Income with Education, GenHlth with PhysHlth), but no correlations are high enough to indicate serious multicollinearity problems.

Q: Which features will you scale, encode, or exclude in preprocessing?  
A: For preprocessing, three main considerations were addressed: scaling, encoding, and exclusion. Several numerical features—BMI, GenHlth, MentHlth, PhysHlth, Age, and Income—were scaled to account for their different ranges and distributions, ensuring they are suitable for algorithms sensitive to feature magnitude. Encoding was not required beyond the dataset’s existing structure: binary variables such as HighBP, HighChol, and Smoker are already represented as 0s and 1s, while ordinal variables like GenHlth, Age, Education, and Income are encoded as integers. No features were excluded, as none contained excessive missing values or were judged irrelevant. Where needed, dimensionality reduction techniques such as PCA can capture the most important variance and mitigate less informative features.

In summary, the main preprocessing adjustment was scaling numerical features, while encoding and exclusion were not necessary at this stage. 

Q: What does your cleaned dataset look like (rows, columns, shape)?  
A: After removing the duplicate rows, our cleaned dataset, stored in the DataFrame df, has the following characteristics:

Shape: The shape of the DataFrame is (229474, 22).
Rows: This means the dataset contains 229,474 rows (which represent individual observations or patients after removing duplicates).
Columns: It has 22 columns (which represent the features and the target variable). 


---
## ✅ Week 2: Feature Engineering & Preprocessing

---

### 🏷️ 1. Feature Encoding

Q: Identify the binary (`0` or `1`) categorical features and apply a simple mapping or encoder. Which features did you encode?

A: The binary columns in the dataset were successfully identified as:

['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex', 'Diabetes_binary']

These features are already encoded as 0 and 1, so no further encoding was required. They were kept as-is for preprocessing and model training.

Q: The `GenHealth` and `Education` features are ordinal. Apply a custom mapping that preserves their inherent order and justify the order you chose. 

A: Both GenHlth and Education are already encoded with values that follow their natural progression. For GenHlth, the scale runs from Excellent (1) to Poor (5), where smaller values represent better health and larger values represent worse health. This order is preserved as-is since it aligns directly with the meaning of the categories. Similarly, Education is coded from 1 = No formal education up to 6 = College graduate, reflecting increasing levels of educational attainment. Because these encodings already respect the ordinal nature of the variables, no remapping is necessary. Retaining the order is preferable to one-hot encoding, as it preserves the meaningful ranking of categories, which allows models to capture thresholds such as “fair or worse health” or “some college or higher.”

Scaling was applied to continuous features such as BMI, MentHlth, and PhysHlth to ensure they are on comparable ranges. Before scaling, these variables had different scales and distributions (e.g., BMI in the 10–100 range, while MentHlth and PhysHlth ranged from 0–30). After applying StandardScaler, each feature was centered around zero with unit variance. This prevents features with larger numerical ranges from dominating model training and improves the performance of algorithms sensitive to feature magnitude (e.g., logistic regression, SVMs, and neural networks). Binary and ordinal features were not scaled, since they already exist in comparable and interpretable units.

Summary

Binary features → already 0/1, no change.

Ordinal features → mapped integers that preserve natural order.

Nominal features → one-hot encoded to avoid artificial hierarchies.

This preprocessing ensures that the dataset’s categorical information is represented accurately and in a way that aligns with machine learning best practices.


Q: For any remaining nominal categorical features, apply one-hot encoding. Why is this method more suitable for nominal data than a simple integer label?  

A:  For nominal categorical features, one-hot encoding is the most appropriate method. Unlike ordinal features, nominal features have no inherent ranking or order. If we were to apply simple integer labels (e.g., assigning 1, 2, 3 to different categories), many machine learning algorithms would incorrectly interpret those numbers as having meaningful magnitude or order (for example, assuming “3 > 2 > 1”), which could bias the model.

One-hot encoding solves this issue by creating a separate binary column for each category, ensuring that all categories are treated as equally distinct. This eliminates the risk of implying false ordinal relationships and allows the model to learn the effect of each category independently.

In summary, one-hot encoding preserves the integrity of nominal data by preventing the introduction of artificial ordering and improving both model accuracy and interpretability.

However, after reviewing the dataset, we found no nominal categorical features requiring one-hot encoding. All categorical variables are either binary (0/1) or ordinal (with a natural numeric order such as GenHlth, Age, or Education). Therefore, no one-hot encoding was applied at this stage. If future datasets introduce text-based or unordered categories (e.g., Race, State), we would use one-hot encoding to avoid imposing an artificial numeric order.---

### ✨ 2. Feature Creation

Q: Create a new feature for BMI categories (e.g., Underweight, Normal, Overweight, Obese) from the `BMI` column. Display the value counts for your new categories.

A:  New Feature: BMI Categories

We engineered a categorical feature BMI_category from the continuous BMI column using CDC guidelines:

Underweight: < 18.5

Normal: 18.5–24.9

Overweight: 25–29.9

Obese: ≥ 30

Distribution across the dataset:

 Category	 Count	  Percent
Overweight	 93,749	   ~37%
Obese	     87,851	   ~35%
Normal	     68,953	   ~27%
Underweight	  3,127	   ~1%
🔍 Insights

The dataset is dominated by Overweight and Obese categories (~72% combined).

This imbalance reinforces the importance of feature engineering and stratification: health risks are not uniformly distributed.

The BMI_category feature may help models capture nonlinear associations between BMI and chronic disease risk.


Q: Create a new feature named `TotalHealthDays` by combining `PhysHlth` and `MentHlth`. What is the rationale behind creating this feature? 

A:  Creating a new feature called TotalHealthDays by summing the number of days in the past month when physical health (PhysHlth) and mental health (MentHlth) were reported as not good.

Rationale:

Both physical and mental health contribute to overall well-being and can impact the risk of diabetes. 
By combining them, we capture a more holistic measure of total “unhealthy days” experienced by an individual in the past 30 days. This feature may help the model recognize patterns where combined poor physical and mental health has a stronger relationship with diabetes outcomes than each measure considered alone.

TotalHealthDays
0     125985
1      11234
2      17388
3      10662
4       6785
5      10182
6       3191
7       5780
8       2119
9       1094
10      6590
11       753
12      1534
13       666
14      2673
Name: count, dtype: int64

By summing these two columns, we get a single metric that represents the total number of days in the past month that a person reported experiencing either physical or mental health issues. This combined feature might provide a more holistic view of a person's health burden and could potentially be a stronger predictor for certain outcomes (like diabetes) than either physical or mental health indicators alone. It allows us to consider the cumulative impact of both aspects of health.


### ✂️ 3. Data Splitting

Q: Split your dataset into training and testing sets (an 80/20 split is recommended). Use stratification on the `Diabetes_binary` target variable.  

A: Now that the data is split, we can proceed with further preprocessing steps on the training and testing sets separately, such as scaling numerical features or encoding categorical features (if needed for the chosen model). Alternatively, we could begin exploring the characteristics of the training and testing datasets. 

Q: Why is it critical to split the data *before* applying techniques like SMOTE or scaling?  
A: It is critical to split your data into training and testing sets before applying techniques like SMOTE (for handling imbalanced data) or scaling (like StandardScaler or MinMaxScaler) to prevent data leakage.

Q: Show the shape of your `X_train`, `X_test`, `y_train`, and `y_test` arrays to confirm the split.  
A: 
Shape of: X_train: (202944, 23)
Shape of X_test: (50736, 23)
Shape of y_train: (202944,)
Shape of y_test: (50736,)
---

### ⚖️ 4. Imbalance Handling & Final Preprocessing

Q: Apply the SMOTE technique to address class imbalance. Importantly, apply it *only* to the training data. Show the class distribution of the training target variable before and after.  
A:  
Class distribution before SMOTE: Counter({0: 174667, 1: 28277})
Class distribution after SMOTE: Counter({0: 174667, 1: 174667})

Shape of X_train_resampled: (349334, 25)
Shape of y_train_resampled: (349334,)


Q: Normalize the numerical features using `StandardScaler`. Fit the scaler *only* on the training data, then transform both the training and testing data. Why must you not fit the scaler on the test data?  
A:  Fitting the StandardScaler (or any other scaler) on the test data would lead to data leakage. The scaler calculates parameters like the mean and standard deviation from the data it's fitted on. If you fit it on the test data, these parameters will be influenced by the distribution of the test set. When you then use this scaler to transform the test data, you are essentially using information from the test set to preprocess the test set itself.


Q: Display the shape of your final, preprocessed training features (`X_train_processed`) and testing features (`X_test_processed`).  
A:
Shape of preprocessed training features (X_train_scaled): (349334, 25)
Shape of preprocessed testing features (X_test_scaled): (91258, 25)


## ✅ Week 3: Model Development & Experimentation

---

### 🤖 1. Baseline Model Training

Q: Which baseline models did you choose for this classification task, and why?  
A:  
The ease of running Naive Bayes, Logistic Regression and Decision Tree, along with metrics to compare.
Given the importance of minimizing false negatives in a medical context like diabetes prediction, the high recall of the Naive Bayes model makes it a strong candidate for further consideration, despite its lower precision. Further tuning of the Logistic Regression model could potentially improve its recall while maintaining its strong performance in accuracy, precision, and AUC.

Q: How did you implement and evaluate Logistic Regression, Decision Tree, and Naive Bayes models?  
A: The predictions from each model were compared to the true labels of the test set (y_test).
The following classification metrics were calculated for each model using functions from sklearn.

Model Performance Comparison:
                       Accuracy	    Precision	Recall	    F1-score	AUC
Naive Bayes	           0.621925	    0.242507	0.806903	0.372932	0.699442
Decision Tree	       0.726683	    0.238014	0.436837	0.308138	0.605221
Logistic Regression    0.666489	    0.258790	0.747630	0.384489	0.700492


Q: What challenges did you encounter during model training, and how did you address them?  
A: Here are the key challenges and how they were addressed:

Non-numerical Data for SMOTE: The SMOTE technique requires numerical input features. Initially, when I attempted to apply SMOTE to X_train, I encountered a ValueError because the BMI_category column contained string values ('Underweight', 'Normal', etc.).
Address: I addressed this by identifying the non-numerical columns (BMI_category in this case) and applying one-hot encoding to them before applying SMOTE. This converted the categorical string values into a numerical format (0s and 1s for the dummy variables) that SMOTE could process.
NaN Values After Scaling and Feature Combination: After applying scaling and attempting to combine the scaled numerical features with the one-hot encoded features for the test set (X_test_scaled), I encountered a ValueError indicating the presence of NaN values. This was likely due to issues with index or column alignment during the concatenation of the scaled numerical and one-hot encoded DataFrames for the test set.

Address: I addressed this by carefully re-generating the one-hot encoded test set (X_test_encoded), ensuring its columns were aligned with the one-hot encoded training set (X_train_encoded) using reindex and filling missing columns with 0. Then, I scaled the original numerical columns of both the encoded training and testing sets separately. Finally, I concatenated the scaled numerical DataFrames with their respective one-hot encoded DataFrames, ensuring correct index alignment using reset_index(drop=True) during concatenation. This process eliminated the NaN values in the final X_train_scaled and X_test_scaled DataFrames.
These preprocessing steps were crucial to ensure that the data was in a suitable format for the classification models and to prevent errors during training and prediction. The errors encountered highlighted the importance of carefully handling categorical features, addressing class imbalance, and correctly applying scaling while avoiding data leakage and NaN values. 

---

### 📈 2. Experiment Tracking

Q: How did you use MLflow (or another tool) to track your experiments?  
A:  

Q: What key parameters and metrics did you log for each model run?  
A:  

Q: How did experiment tracking help you compare and select the best model?  
A:  

---

### 🧮 3. Model Evaluation

Q: Which evaluation metrics did you use to assess model performance, and why are they appropriate for this problem?  
A:  

Q: How did you interpret the accuracy, precision, recall, and F1-score for your models?  
A:  

Q: Did you observe any trade-offs between different metrics? How did you decide which metric(s) to prioritize?  
A:  

---

### 🕵️ 4. Error Analysis

Q: How did you use confusion matrices to analyze model errors?  
A:  

Q: What types of misclassifications were most common, and what might explain them?  
A:  

Q: How did your error analysis inform your next steps in model improvement?  
A:  

---

### 📝 5. Model Selection & Insights

Q: Based on your experiments, which model performed best and why?  
A:  

Q: What are your top 3–5 insights from model development and experimentation?  
A:  

Q: How would you communicate your model’s strengths and limitations to a non-technical stakeholder?  
A:  

---

## ✅ Week 4: Model Tuning & Finalization

---

### 🛠️ 1. Hyperparameter Tuning

Q: Which hyperparameters did you tune for your models, and what methods (e.g., grid search, random search) did you use?  
A:  

Q: How did you select the range or values for each hyperparameter?  
A:  

Q: What impact did hyperparameter tuning have on your model’s performance?  
A:  

---

### 🔄 2. Cross-Validation

Q: How did you use cross-validation to assess model stability and generalization?  
A:  

Q: What were the results of your cross-validation, and did you observe any variance across folds?  
A:  

Q: Why is cross-validation important in this context?  
A:  

---

### 🏆 3. Final Model Selection

Q: How did you choose your final model after tuning and validation?  
A:  

Q: Did you retrain your final model on the full training set before evaluating on the test set? Why or why not?  
A:  

Q: What were the final test set results, and how do they compare to your validation results?  
A:  

---

### 📊 4. Feature Importance & Interpretation

Q: How did you assess feature importance for your final model?  
A:  

Q: Which features were most influential in predicting diabetes risk, and do these results align with domain knowledge?  
A:  

Q: How would you explain your model’s decision process to a non-technical audience?  
A:

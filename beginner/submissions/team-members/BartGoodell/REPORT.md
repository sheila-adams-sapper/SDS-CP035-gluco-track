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

### 🏷️ 1. Feature Encoding

Q: Identify the binary (`0` or `1`) categorical features and apply a simple mapping or encoder. Which features did you encode?  
A:  
    Diabetes_binary   Target   Binary             None   
2                 HighBP  Feature   Binary             None   
3               HighChol  Feature   Binary             None   
4              CholCheck  Feature   Binary             None   
6                 Smoker  Feature   Binary             None   
7                 Stroke  Feature   Binary             None   
8   HeartDiseaseorAttack  Feature   Binary             None   
9           PhysActivity  Feature   Binary             None   
10                Fruits  Feature   Binary             None   
11               Veggies  Feature   Binary             None   
12     HvyAlcoholConsump  Feature   Binary             None   
13         AnyHealthcare  Feature   Binary             None   
14           NoDocbcCost  Feature   Binary             None   
18              DiffWalk  Feature   Binary             None   
19                   Sex  Feature   Binary              Sex 


Q: The `GenHealth` and `Education` features are ordinal. Apply a custom mapping that preserves their inherent order and justify the order you chose. 

A: The dataset description for 'GenHlth' indicates that the integer values represent a subjective assessment of general health, where 1 is Excellent, 2 is Very Good, 3 is Good, 4 is Fair, and 5 is Poor. The chosen mapping preserves this inherent ordinal structure, assigning a numerical value to each level that reflects its position on a scale from best health (1) to worst health (5).

The dataset description for 'Education' indicates that the integer values represent increasing levels of education, from no schooling (1) to college graduate (6). The chosen mapping preserves this inherent ordinal structure, assigning a numerical value to each level that reflects its position in the hierarchy of educational attainment

Q: For any remaining nominal categorical features, apply one-hot encoding. Why is this method more suitable for nominal data than a simple integer label?  
A:  No nominal features were identified in the dataset at the stage where one-hot encoding was considered, so no features were one-hot encoded.
---

### ✨ 2. Feature Creation

Q: Create a new feature for BMI categories (e.g., Underweight, Normal, Overweight, Obese) from the `BMI` column. Display the value counts for your new categories.  
A:  

Q: Create a new feature named `TotalHealthDays` by combining `PhysHlth` and `MentHlth`. What is the rationale behind creating this feature?  
A:  

---

### ✂️ 3. Data Splitting

Q: Split your dataset into training and testing sets (an 80/20 split is recommended). Use stratification on the `Diabetes_binary` target variable.  
A:  

Q: Why is it critical to split the data *before* applying techniques like SMOTE or scaling?  
A:  

Q: Show the shape of your `X_train`, `X_test`, `y_train`, and `y_test` arrays to confirm the split.  
A:  

---

### ⚖️ 4. Imbalance Handling & Final Preprocessing

Q: Apply the SMOTE technique to address class imbalance. Importantly, apply it *only* to the training data. Show the class distribution of the training target variable before and after.  
A:  

Q: Normalize the numerical features using `StandardScaler`. Fit the scaler *only* on the training data, then transform both the training and testing data. Why must you not fit the scaler on the test data?  
A:  

Q: Display the shape of your final, preprocessed training features (`X_train_processed`) and testing features (`X_test_processed`).  
A:

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
A: I used MLflow to track my experiments. For each algorithm I trained, MLflow automatically recorded parameters, metrics, and outputs. These runs were logged into the mlruns directory, where I could compare results across models. This allowed me to keep a clear record of which algorithms were tested, how they performed, and to reference the results consistently in my analysis.

Q: What key parameters and metrics did you log for each model run?  
A:  
 index	           Accuracy  Precision	 Recall	 F1-score   AUC
Naive Bayes  	    0.644 	  0.256	      0.813	  0.389	   0.715
Decision Tree	    0.750	  0.254	      0.412	  0.314	   0.608
Logistic Regression	0.727	  0.297	      0.701	  0.417	   0.716
Random Forest	    0.780	  0.308       0.466	  0.371	   0.648
Gradient Boosting	0.720	  0.295	      0.730   0.420	   0.724
k-Nearest Neighbors	0.720	  0.268	      0.583   0.367	   0.663
Summary:
Comparing all six models:
- Random Forest performs best in terms of Accuracy (0.7800).
- Random Forest performs best in terms of Precision (0.3083), which is important for minimizing false positives.
- Naive Bayes performs best in terms of Recall (0.8127), which is crucial for minimizing false negatives in diabetes prediction.
- Gradient Boosting achieves the highest F1-score (0.4203), indicating the best balance between precision and recall.
- Gradient Boosting has the highest AUC (0.7238), showing the best overall ability to distinguish between classes.

Performance of new models compared to initial models:
Random Forest shows good overall performance, particularly in Accuracy and Precision, and is competitive with Logistic Regression. However, its Recall is lower than Naive Bayes and Gradient Boosting.
Gradient Boosting demonstrates strong Recall, second only to Naive Bayes, and has a competitive F1-score and AUC. It appears to be a good model for identifying positive cases.
k-Nearest Neighbors generally performs less well across most metrics compared to the ensemble methods (Random Forest and Gradient Boosting) and Logistic Regression, although it has a higher Recall than Decision Tree and Random Forest.
Overall, considering the importance of Recall in this medical context, Naive Bayes and Gradient Boosting show strengths in identifying diabetes cases. Logistic Regression and Random Forest offer a better balance of precision and accuracy.
Distributions.

Q: How did experiment tracking help you compare and select the best model?  
A: Centralized recording of the mlruns where details are logged in one file and can be compared easily side by side. Note above table. Easy comparison of metrics from run to run. Understanding of parameters and see how different parameters affect the model performances from run to run and this will help to find the optimal settings. Also, reproducibility from run to run and the ability to find specific tuning parameters to adjust or reevaluate if necessary. 

Q: Which evaluation metrics did you use to assess model performance, and why are they appropriate for this problem?  
A: I used accuracy, precision, recall, F1-score, and AUC to assess model performance.
Accuracy gave a general sense of overall performance, but by itself is misleading due to the class imbalance. 

Precision measured how many predicted positive cases (diabetes) were correct, helping to assess the risk of false alarms.

Recall measured how many true diabetes cases the model correctly identified, which is critical in a healthcare context where missed diagnoses (false negatives) are especially costly.

F1-score balanced precision and recall into a single metric, useful for comparing models in this imbalanced setting.

AUC (Area Under the ROC Curve) provided an overall measure of the model’s ability to discriminate between diabetic and non-diabetic cases across thresholds.

These metrics are appropriate because they highlight not just how often the model is right, but the types of errors it makes—a crucial distinction when the consequences of false negatives and false positives differ.

Precision: The proportion of correctly predicted positive instances (true positives) out of all instances predicted as positive (true positives + false positives). It measures the model's ability to avoid false positives.
Why it's appropriate: In the context of diabetes prediction, a false positive would mean predicting that someone has diabetes when they actually don't. High precision is important to minimize unnecessary follow-up tests, anxiety, and costs associated with false diagnoses.

Recall (Sensitivity): The proportion of correctly predicted positive instances (true positives) out of all actual positive instances (true positives + false negatives). It measures the model's ability to find all the positive instances. High recall is essential to minimize false negatives and ensure that as many actual diabetes cases as possible are identified.

F1-score: The harmonic mean of Precision and Recall. It provides a single score that balances both precision and recall. The F1-score is particularly useful when dealing with imbalanced datasets. It gives a better measure of the model's performance than accuracy alone because it considers both false positives and false negatives, providing a balanced evaluation of the model's ability to classify the positive class.

AUC (Area Under the Receiver Operating Characteristic Curve): A measure of the model's ability to distinguish between the positive and negative classes across all possible classification thresholds. An AUC of 1.0 represents a perfect classifier, while an AUC of 0.5 represents a random classifier. AUC is a robust metric for imbalanced datasets because it evaluates the model's performance across various thresholds and is not influenced by the class distribution. It provides an overall measure of the classifier's discriminatory power.


Q: How did you interpret the accuracy, precision, recall, and F1-score for your models?  
A: While the models achieved high accuracy, this could largely reflect correct predictions of the majority class. Precision helped us see how many of the predicted positive cases were actually diabetic, while recall revealed how many true diabetic cases were successfully identified. Given the healthcare context, recall is particularly important to minimize false negatives (missed diabetic cases), though precision also matters to avoid unnecessary false alarms. The F1-score provided a balanced measure between precision and recall, giving us a clearer picture of overall model performance on this imbalanced dataset.

Q: Did you observe any trade-offs between different metrics? How did you decide which metric(s) to prioritize?  
A: In imbalanced classification problems, precision and recall often trade off. High precision means fewer false positives but may miss true cases (lower recall), while high recall catches more true positives but risks more false alarms. In our results, for example, models like Naive Bayes leaned toward higher recall but lower precision, while others did the opposite.

For diabetes prediction, recall is especially critical because a false negative (missing a diagnosis) could delay treatment and harm patients. At the same time, precision still matters to avoid unnecessary testing and anxiety. The F1-score balances these two, and AUC shows overall discriminatory power across thresholds. In practice, a model with strong recall, reasonable precision, and a solid F1/AUC is preferred.

---

### 🕵️ 4. Error Analysis

Q: How did you use confusion matrices to analyze model errors?  
A: A confusion matrix summarizes classification performance by showing counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). For diabetes prediction, these correspond to correctly or incorrectly predicting whether someone has diabetes.

The matrix is valuable because it reveals the types of errors the model makes, not just overall accuracy. For example:

False Negatives (FN): Missed diabetes cases — the most concerning error in this medical context.

False Positives (FP): Predicting diabetes when the person is healthy — less harmful than FN but still costly.

From the confusion matrix we can directly calculate metrics like precision, recall, and F1-score, but the visual breakdown makes it clearer if a model is skewed toward one type of error. For instance, even a model with high accuracy might still show many false negatives, which would be unacceptable for healthcare screening.


Q: What types of misclassifications were most common, and what might explain them?  
A: Misclassification Summary (Logistic Regression) The confusion matrix showed more false positives (11,728) than false negatives (2,116). False positives likely stem from class imbalance, overlapping risk factors (e.g., BMI, blood pressure), and the linear limits of Logistic Regression. False negatives, while fewer, remain critical since they represent missed diagnoses. In practice, recall should be prioritized to avoid missed cases, but balancing precision and recall (e.g., via threshold tuning or more complex models) is key.   

Q: How did your error analysis inform your next steps in model improvement?   
A: Gradient Boosting, Logistic Regression, and Random Forest appear to offer the best chances for significant performance improvement through tuning, given their current performance and the range of hyperparameters available to optimize. Focusing tuning efforts on Gradient Boosting, which showed a good balance with the highest F1-score and AUC, could be a promising next step to achieve optimal results for this problem.


### 📝 5. Model Selection & Insights

Q: Based on your experiments, which model performed best and why?  
A: Gradient Boosting, Logistic Regression, and Random Forest appear to offer the best chances for significant performance improvement through tuning, given their current performance and the range of hyperparameters available to optimize. Focusing tuning efforts on Gradient Boosting, which showed a good balance with the highest F1-score and AUC, could be a promising next step to achieve optimal results for this problem.



Q: What are your top 3–5 insights from model development and experimentation?  
A:  Class imbalance drives results: With far fewer diabetes cases, accuracy is misleading on its own. Using SMOTE in training helped the model learn minority patterns, but evaluation on the original test set required metrics like Precision, Recall, F1, and AUC.

Metric trade-offs: Models that maximize Recall (catching more diabetics) often suffer lower Precision (more false alarms), and vice versa. In healthcare, Recall is critical to minimize missed cases, though Precision still matters.

Ensembles perform best: Gradient Boosting and Random Forest provided the best balance of Precision, Recall, and overall AUC, outperforming simpler models like Decision Tree and Naive Bayes.

Feature engineering adds value: Derived features like BMI_category and TotalHealthDays improved model interpretability and captured patterns raw features missed.

Tuning opportunities remain: Complex models (Gradient Boosting, Random Forest, Logistic Regression) showed strong results and could likely be improved further with hyperparameter tuning.

Q: How would you communicate your model’s strengths and limitations to a non-technical stakeholder?  
A: Our model is designed to help flag individuals who may be at risk of diabetes based on health and lifestyle factors. Its main strength is that it’s very good at identifying people who are likely to have diabetes (high recall), which means we reduce the chances of missing someone who actually needs care. This is especially important in a health context, because a missed diagnosis can be far more harmful than a false alarm.

That said, the model does sometimes predict diabetes when the person is actually healthy (false positives). While these cases don’t carry medical harm, they could lead to unnecessary follow-up testing or anxiety. This trade-off is common in screening tools, and we can adjust how “cautious” the model is depending on the priorities (catch every possible case vs. avoid false alarms).

In summary:

Strengths: Helps catch more true diabetes cases early; uses health features that are easy to collect; models complex patterns effectively.

Limitations: Some healthy individuals may be flagged unnecessarily; results depend on the quality and granularity of the input data; further tuning and testing would be needed before real-world deployment.

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

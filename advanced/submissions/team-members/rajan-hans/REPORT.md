# 🔴 GlucoTrack – Advanced Track

## ✅ Week 1: Exploratory Data Analysis (EDA)

---

### 📦 1. Data Integrity & Structure

**Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?**  
**A:** No missing values were detected for any column (all columns show 0 missing). The notebook also reported **0 duplicate rows**. Column value ranges and sample previews matched the documented codebook expectations, so no incorrectly formatted entries were observed.

**Q: Are all data types appropriate (e.g., numeric, categorical)?**  
**A:** Yes. Binary indicators are encoded as 0/1 integers (e.g., `HighBP`, `HighChol`, `Smoker`, etc.). Ordinal categories appear as small integer buckets (e.g., `GenHlth` 1–5, `Education` 1–6, `Income` 1–8, `Age` 1–13). Count/continuous-like features (`BMI`, `MentHlth`, `PhysHlth`) are numeric. This aligns with the dataset’s schema and supports both classical ML and NN pipelines.

**Q: Did you detect any constant, near-constant, or irrelevant features?**  
**A:** **No constant features** were found. Unique-value counts show at least two levels for all predictors. `ID` is analytically irrelevant and will be excluded from modeling to prevent leakage.

---

### 🎯 2. Target Variable Assessment

**Q: What is the distribution of `Diabetes_binary`?**  
**A:** The distribution shows: **Non‑Diabetes: 86.0667%**, **Pre-diabetes/Diabetes: 13.9333%**.

**Q: Is there a class imbalance? If so, how significant is it?**  
**A:** Yes—substantial. The negative:positive ratio is approximately **6.18 : 1**, indicating a minority positive class (~14%).

**Q: How might this imbalance influence your choice of evaluation metrics or model strategy?**  
**A:** Accuracy alone would be misleading. We will prioritize **PR‑AUC** (baseline equals the positive prevalence), report **ROC‑AUC**, and track **recall** at clinically reasonable false‑positive rates. During training we’ll use **class weighting** (and consider focal loss / threshold tuning). Probability **calibration** will be checked before deployment.

---

### 📊 3. Feature Distribution & Quality

**Q: Which numerical features are skewed or contain outliers?**  
**A:** The histograms/boxplots indicate **right‑skew** and visible high-end outliers for **`BMI`**, **`PhysHlth`**, and **`MentHlth`**.

**Q: Did any features contain unrealistic or problematic values?**  
**A:** None observed. Ordinal features stayed within documented ranges (e.g., `GenHlth` 1–5; `Age` 1–13; `Education` 1–6; `Income` 1–8). `MentHlth`/`PhysHlth` counts ranged 0–30 as expected. `BMI` shows high values/outliers but nothing flagged as invalid in the notebook outputs.

**Q: What transformation methods (if any) might improve these feature distributions?**  
**A:** For linear/NN models we’ll **standardize** key numerics (`BMI`, `PhysHlth`, `MentHlth`). If tails remain influential, we’ll consider **winsorization** or a **Yeo‑Johnson/Quantile** transform on those features. Tree models generally handle skew without transformation.

---

### 📈 4. Feature Relationships & Patterns

**Q: Which categorical features (e.g., `GenHlth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?**  
**A:** The “positive‑rate by category” plots show clear signal for:  
- **`GenHlth`**: prevalence increases monotonically from *Excellent* to *Poor*.  
- **`PhysActivity`**: active individuals show a lower positive rate than inactive.  
- **`HighBP` / `HighChol`**: groups with value 1 exhibit noticeably higher positive rates.  
- **Age buckets**: older age groups show higher prevalence.

**Q: Are there any strong pairwise relationships or multicollinearity between features?**  
**A:** The correlation heatmap highlighted a strong pair: **`GenHlth`–`PhysHlth` ≈ 0.524** (|ρ|>0.5). Several other health‑status variables showed moderate associations, so we’ll monitor multicollinearity (especially for linear baselines) and consider dropping/combining redundant features if needed.

**Q: What trends or correlations stood out during your analysis?**  
**A:** Worse self‑reported health, cardiometabolic indicators (`HighBP`, `HighChol`), and inactivity aligned with higher diabetes prevalence; age also showed an increasing trend. `BMI` displayed a positive association with the target in stratified views.

---

### 🧰 5. EDA Summary & Preprocessing Plan

**Q: What are your 3–5 biggest takeaways from EDA?**  
**A:**  
1) **Class imbalance** is significant (~14% positive), so metric choice and training strategy must account for it.  
2) **Health‑status and behavior variables** (`GenHlth`, `PhysActivity`, `HighBP`, `HighChol`, `BMI`, `Age`) carry strong signal.  
3) **Skewed numerics** (`BMI`, `PhysHlth`, `MentHlth`) warrant scaling and possibly robust transforms.  
4) **No missing data** and **no duplicates** detected; the table is analysis‑ready.  
5) **Some collinearity** among health status variables (e.g., `GenHlth` with `PhysHlth`) should be managed for linear models.

**Q: Which features will you scale, encode, or exclude in preprocessing?**  
**A:**  
- **Scale**: `BMI`, `PhysHlth`, `MentHlth` (standardization; consider robust transform if needed).  
- **Encode**: Treat `GenHlth`, `Age`, `Education`, `Income` as **ordinal** for classical models; for NNs, use **integer indices + embeddings**. Binary flags remain as 0/1.  
- **Exclude**: `ID` (non‑predictive identifier).

**Q: What does your cleaned dataset look like (rows, columns, shape)?**  
**A:** Notebook reports **(253,680 rows × 24 columns)** with 0 missing and 0 duplicates. For modeling, we will **drop `ID`**, yielding **23 features + 1 target** (24 columns total during training artifacts).



## ✅ Week 2: Feature Engineering & Deep Learning Prep

---

### 🏷️ 1. Categorical Feature Encoding

**Q: Which categorical features in the dataset have more than two unique values?**
**A:**  As we have seen these are the High-cardinality categorical features (more than 2 unique values): ['GenHlth', 'Age', 'Education', 'Income']

**Q: Apply integer-encoding to these high-cardinality features. Why is this strategy suitable for a subsequent neural network with an embedding layer?**
**A:**  The high-cardinality features are already integer-encoded as below:
| Feature   |   Value | Meaning                                              |
|:----------|--------:|:-----------------------------------------------------|
| GenHlth   |       1 | Excellent                                            |
| GenHlth   |       2 | Very Good                                            |
| GenHlth   |       3 | Good                                                 |
| GenHlth   |       4 | Fair                                                 |
| GenHlth   |       5 | Poor                                                 |
| Age       |       1 | 18-24                                                |
| Age       |       2 | 25-29                                                |
| Age       |       3 | 30-34                                                |
| Age       |       4 | 35-39                                                |
| Age       |       5 | 40-44                                                |
| Age       |       6 | 45-49                                                |
| Age       |       7 | 50-54                                                |
| Age       |       8 | 55-59                                                |
| Age       |       9 | 60-64                                                |
| Age       |      10 | 65-69                                                |
| Age       |      11 | 70-74                                                |
| Age       |      12 | 75-79                                                |
| Age       |      13 | 80 or older                                          |
| Education |       1 | Never attended school or only kindergarten           |
| Education |       2 | Grades 1-8 (Elementary)                              |
| Education |       3 | Grades 9-11 (Some high school)                       |
| Education |       4 | Grade 12 or GED (High school graduate)               |
| Education |       5 | College 1-3 years (Some college or technical school) |
| Education |       6 | College 4 years or more (College graduate)           |
| Income    |       1 | Less than $10,000                                    |
| Income    |       2 | $10,000 to less than $15,000                         |
| Income    |       3 | $15,000 to less than $20,000                         |
| Income    |       4 | $20,000 to less than $25,000                         |
| Income    |       5 | $25,000 to less than $35,000                         |
| Income    |       6 | $35,000 to less than $50,000                         |
| Income    |       7 | $50,000 to less than $75,000                         |
| Income    |       8 | $75,000 or more                                      |
|:----------|--------:|:-----------------------------------------------------
**Q: Display the first 5 rows of the transformed data to show the new integer labels.**  
**A:**  First 5 rows of transformed data with integer labels:
   bmi_category_encoded  Age_encoded  GenHlth_encoded
0  5                     8            4              
1  2                     6            2              
2  2                     8            4              
3  2                     10           1              
4  1                     10           1              

---

### ⚖️ 2. Numerical Feature Scaling

**Q: Which numerical features did your EDA from Week 1 suggest would benefit from scaling?**  
A:  Based on Week 1 EDA findings: MentHlth and PhysHlth show right-skewed distributions and would benefit from scaling

**Q: Apply a scaling technique to these features. Justify your choice of `StandardScaler` vs. `MinMaxScaler` or another method.**  
A:  StandardScaler was used instead of MinMaxScaler because it standardizes features by removing the mean and scaling to unit variance (mean = 0, std = 1). This is ideal for features like MentHlth and PhysHlth that have right-skewed distributions and may not be naturally bounded. Standardization helps many machine learning algorithms (especially those that assume Gaussian-like distributions or are sensitive to feature scale, such as neural networks and linear models) to converge faster and perform better.

MinMaxScaler rescales features to a fixed range (usually 0 to 1), which is useful when you want all features strictly within a bounded interval, but it can be sensitive to outliers and does not center the data. In this notebook, the goal was to normalize the distribution and variance of the selected features, making StandardScaler the more appropriate choice.

**Q: Show the summary statistics of the scaled data to confirm the transformation was successful.**
A:  Summary statistics after scaling:
           MentHlth      PhysHlth
count  2.294740e+05  2.294740e+05
mean  -2.675287e-17  2.972542e-17
std    1.000002e+00  1.000002e+00
min   -4.547857e-01 -5.172127e-01
25%   -4.547857e-01 -5.172127e-01
50%   -4.547857e-01 -5.172127e-01
75%   -1.956387e-01 -7.526566e-02
max    3.432420e+00  2.797390e+00


### ✂️ 3. Stratified Data Splitting

**Q: Split the data into training, validation, and testing sets (e.g., 70/15/15). What function and parameters did you use?**
A:  The function used is train_test_split from scikit-learn. It is used to split arrays or dataframes into random train and test subsets. In your code, it is used to:

Split the feature set (X) and target (y) into two parts: a test set (15% of the data) and a temporary set (85% of the data).
The stratify=y argument ensures that the class distribution (e.g., proportion of diabetes cases) is preserved in both splits.
The random_state=42 argument ensures reproducibility of the split.
This function is essential for creating unbiased training, validation, and test sets for machine learning, especially when dealing with imbalanced classes. It helps evaluate model performance on unseen data and prevents overfitting.

**Q: Why is it critical to use stratification for this specific dataset?**
A:  Stratification ensures that the class distribution (e.g., proportion of diabetes cases) is preserved in both splits.

**Q: Verify the stratification by showing the class distribution of `Diabetes_binary` in each of the three resulting sets.**
A:  Stratification was successful. The plots show class balance is maintained across all splits

---

### 📦 4. Deep Learning Dataset Preparation

**Q: Convert your three data splits into PyTorch `DataLoader` or TensorFlow `tf.data.Dataset` objects. What batch size did you choose and why?**
A:  Converted the three data splits - Traon, Val and Test into pytorch dataloaders
Batch size was kept as 64
A batch size of 64 is commonly used because it offers a good balance between training speed and model performance. Larger batch sizes (like 64) allow for more efficient computation on modern hardware (especially GPUs), as more data is processed in parallel, which can speed up training. It also helps stabilize gradient estimates, making optimization smoother.

A batch size of 32 is also popular and may sometimes lead to slightly better generalization, but it processes fewer samples per step, which can make training slower. The choice between 32 and 64 is not strict—both are reasonable defaults. In practice, the best batch size depends on your dataset size, available memory, and hardware. For this notebook, 64 was chosen as a typical value for deep learning tasks, but you can experiment with 32 or other values to see what works best for your setup.

**Q: To confirm they are set up correctly, retrieve one batch from your training loader. What is the shape of the features (X) and labels (y) in this batch?**
A:  The features (X) in one batch have shape [64, 22].
The labels (y) in one batch have shape [64, 1].

**Q: Explain the role of the `shuffle` parameter in your training loader. Why is this setting important for the training set but not for the validation or testing sets?**
A:The shuffle parameter in your training DataLoader randomizes the order of samples in each epoch. This is important for the training set because it prevents the model from learning patterns based on the order of the data, improves generalization, and helps avoid overfitting. Shuffling ensures that each batch contains a diverse mix of samples, which leads to more stable and effective gradient updates.

For validation and testing sets, shuffling is not needed because you want to evaluate the model on a fixed, consistent set of data. Keeping the order unchanged ensures reproducible and reliable performance metrics.
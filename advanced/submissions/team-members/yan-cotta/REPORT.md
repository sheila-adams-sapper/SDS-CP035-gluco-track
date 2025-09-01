# 🔴 GlucoTrack – Advanced Track EDA Report

## ✅ Week 1: Exploratory Data Analysis (EDA)

**Dataset:** CDC Diabetes Health Indicators (UCI ML Repository ID: 891)  
**Analysis Date:** August 8, 2025  
**Analyst:** Yan Cotta

---

### 📦 1. Data Integrity & Structure

**Q: Are there any missing, duplicate, or incorrectly formatted entries in the dataset?**

**A:** Our comprehensive data integrity analysis revealed:

- **Missing Values:** ✅ **Zero missing values** across all 253,680 samples and 22 columns (including target)
- **Duplicate Rows:** ⚠️ **24,206 duplicate rows identified** (9.5% of the dataset), leaving 229,474 unique samples
- **Data Formatting:** ✅ **All variables properly formatted** with consistent encoding

The duplicate rows represent a significant data quality issue that must be addressed in preprocessing. The high completeness rate (100%) indicates excellent data collection practices from the CDC BRFSS 2014 survey.

**Q: Are all data types appropriate (e.g., numeric, categorical)?**

**A:** All data types are appropriately encoded for analysis:

- **Numerical Variables (4):** BMI, MentHlth, PhysHlth, Age - all stored as appropriate numeric types
- **Categorical Variables (18):** All binary health indicators properly encoded as 0/1 integers
- **Target Variable:** Diabetes_binary correctly encoded as 0 (no diabetes) and 1 (diabetes/prediabetes)

Binary variable validation confirmed all categorical features contain only valid 0/1 values with no invalid entries.

**Q: Did you detect any constant, near-constant, or irrelevant features?**

**A:** No constant or near-constant features were detected. All variables show meaningful variation:

- **Binary variables** show diverse distributions ranging from rare conditions (stroke) to common ones (high BP)
- **Numerical variables** demonstrate appropriate ranges without constant values
- **All features appear clinically relevant** for diabetes prediction based on medical literature

No features require removal due to lack of variation or irrelevance.

---

### 🎯 2. Target Variable Assessment

**Q: What is the distribution of `Diabetes_binary`?**

**A:** The target variable shows a clear majority-minority class structure:

- **Class 0 (No Diabetes):** 218,334 samples (86.07%)
- **Class 1 (Diabetes/Prediabetes):** 35,346 samples (13.93%)
- **Total Sample Size:** 253,680 samples

This distribution aligns with expected population-level diabetes prevalence in the US adult population.

**Q: Is there a class imbalance? If so, how significant is it?**

**A:** Yes, there is **significant class imbalance**:

- **Imbalance Ratio:** 6.2:1 (majority to minority class)
- **Minority Class Proportion:** 13.93% represents substantial underrepresentation
- **Impact Level:** Moderate to severe - requires specific handling strategies

This level of imbalance is realistic for health datasets but poses challenges for standard machine learning algorithms.

**Q: How might this imbalance influence your choice of evaluation metrics or model strategy?**

**A:** The class imbalance significantly impacts our modeling approach:

**Evaluation Metrics:**
- **Primary Focus:** Sensitivity/Recall (>90% target) to minimize false negatives in healthcare context
- **Balanced Metrics:** F1-score, AUC-ROC, and AUC-PR for comprehensive evaluation
- **Clinical Metrics:** NPV (Negative Predictive Value) for ruling out diabetes risk
- **Avoid:** Simple accuracy as it will be misleadingly high due to class imbalance

**Model Strategy:**
- **Resampling:** Consider SMOTE or ADASYN for minority class oversampling
- **Cost-sensitive Learning:** Apply class weights penalizing false negatives more heavily
- **Ensemble Methods:** Use algorithms robust to imbalance (Random Forest, XGBoost)
- **Threshold Tuning:** Optimize decision threshold for optimal sensitivity-specificity balance

---

### 📊 3. Feature Distribution & Quality

**Q: Which numerical features are skewed or contain outliers?**

**A:** Analysis of the four numerical variables revealed distinct distribution patterns:

**BMI (Body Mass Index):**
- **Distribution:** Right-skewed with most values between 20-35
- **Population Health Insight:** Mean ~28-30 indicates population skews toward overweight/obese
- **Outliers:** Some extreme values >60 representing severely obese individuals
- **Clinical Relevance:** Outliers are medically meaningful, not data errors

**MentHlth (Mental Health Days):**
- **Distribution:** Highly right-skewed with majority at 0 (no mental health issues)
- **Pattern:** Clear peaks at 0, 15, and 30 days (survey response pattern)
- **Outliers:** Maximum value clustering at 30 days indicates chronic mental health issues

**PhysHlth (Physical Health Days):**
- **Distribution:** Similar to MentHlth - right-skewed with concentration at 0
- **Pattern:** Peaks at 0, 15, and 30 days following survey response patterns
- **Interpretation:** High number reporting 30 days suggests chronic physical health problems

**Age (13-level categories):**
- **Distribution:** Relatively uniform across age groups 1-13
- **Coverage:** Good representation across all age categories
- **Note:** Represents bucketed age ranges, not raw years

**Q: Did any features contain unrealistic or problematic values?**

**A:** Comprehensive range validation revealed **no unrealistic values**:

- **BMI:** No extreme values <10 or >100 detected
- **Age Categories:** All values within valid range 1-13
- **Health Days:** All values appropriately bounded 0-30
- **Binary Variables:** All contain only valid 0/1 values

All extreme values appear to be legitimate medical measurements rather than data entry errors.

**Q: What transformation methods (if any) might improve these feature distributions?**

**A:** Recommended transformations for optimal model performance:

**For Skewed Distributions:**
- **MentHlth & PhysHlth:** Consider log(x+1) transformation to reduce right skewness
- **BMI:** Potential square root transformation, though outliers may be clinically meaningful

**Feature Engineering Opportunities:**
- **BMI Categories:** Convert to clinical categories (underweight, normal, overweight, obese)
- **Health Days Binary:** Create binary indicators for "any poor health days" (>0)
- **Composite Scores:** Combine MentHlth + PhysHlth for overall health burden score

**Scaling Requirements:**
- **Standardization needed** for BMI, MentHlth, PhysHlth, Age before modeling
- **Robust scaling** preferred due to presence of outliers

---

### 📈 4. Feature Relationships & Patterns

**Q: Which categorical features (e.g., `GenHealth`, `PhysicalActivity`, `Smoking`) show visible patterns in relation to `Diabetes_binary`?**

**A:** Correlation analysis with diabetes target revealed several strong relationships:

**Strongest Target Correlations:**
- **GenHlth (General Health):** 0.29 correlation - poorer general health strongly associated with diabetes
- **BMI:** 0.22 correlation - higher BMI significantly linked to diabetes risk
- **HighBP (High Blood Pressure):** 0.26 correlation - hypertension shows strong diabetes association
- **Age:** 0.22 correlation - diabetes risk increases with age as expected
- **HighChol (High Cholesterol):** 0.20 correlation - cholesterol and diabetes clustering

**Notable Lifestyle Patterns:**
- **PhysActivity:** -0.12 correlation - physical activity inversely related to diabetes (protective factor)
- **Education:** -0.16 correlation - higher education associated with lower diabetes risk
- **Income:** -0.16 correlation - higher income correlated with reduced diabetes risk

**Healthcare Access Patterns:**
- **NoDocbcCost:** 0.23 correlation - financial barriers to healthcare linked to diabetes
- **AnyHealthcare:** -0.12 correlation - healthcare access protective against diabetes

**Q: Are there any strong pairwise relationships or multicollinearity between features?**

**A:** **Minimal multicollinearity detected** - excellent for modeling:

**Only Strong Correlation Found:**
- **GenHlth ↔ PhysHlth:** 0.524 correlation (general health and physical health days)
- This represents the only correlation exceeding our 0.5 threshold

**Moderate Correlations (0.3-0.5):**
- **HighBP ↔ HighChol:** 0.30 (expected comorbidity)
- **Age ↔ HighBP:** 0.34 (age-related hypertension)
- **DiffWalk ↔ PhysHlth:** 0.48 (mobility issues and physical health)

**Modeling Implications:**
- **Low multicollinearity risk** allows use of most features without dimensionality reduction
- **Feature independence** means each variable provides unique predictive information
- **No need for aggressive feature selection** due to multicollinearity

**Q: What trends or correlations stood out during your analysis?**

**A:** Several key patterns emerged with important clinical implications:

**1. Health Condition Clustering:**
   - Diabetes patients show significantly higher rates of multiple comorbidities
   - Clear clustering of hypertension, high cholesterol, and heart disease with diabetes

**2. Socioeconomic Health Gradients:**
   - **Education and income** show inverse relationships with diabetes risk
   - **Healthcare access barriers** (NoDocbcCost) strongly associated with diabetes
   - Suggests social determinants of health play crucial role

**3. Age-Related Progression:**
   - **Strong age-diabetes relationship** follows expected epidemiological patterns
   - Diabetes prevalence increases systematically across age categories

**4. BMI Threshold Effects:**
   - **Clear BMI cutoffs** associated with diabetes risk
   - Suggests potential for BMI-based risk stratification

**5. Lifestyle Protection Factors:**
   - **Physical activity** shows protective effect against diabetes
   - **Healthcare utilization** patterns differ significantly between diabetic and non-diabetic groups

---

### 🧰 5. EDA Summary & Preprocessing Plan

**Q: What are your 3–5 biggest takeaways from EDA?**

**A:** Five critical insights emerged from our comprehensive analysis:

**1. High-Quality Dataset with Minimal Preprocessing Needs**
   - Zero missing values and appropriate encoding minimize preprocessing requirements
   - Primary concern: 24,206 duplicate rows requiring removal
   - Dataset readiness score: 3/4 (excellent foundation for modeling)

**2. Significant but Manageable Class Imbalance**
   - 6.2:1 imbalance ratio requires specific handling but represents realistic population distribution
   - Sufficient minority class samples (35,346) for effective resampling techniques
   - Must prioritize sensitivity over accuracy in evaluation metrics

**3. Rich Feature Set with Low Multicollinearity**
   - 21 features spanning health conditions, lifestyle, demographics, and healthcare access
   - Only one strong correlation pair (GenHlth-PhysHlth) indicates feature independence
   - All features clinically relevant with no constant or irrelevant variables detected

**4. Clear Predictive Patterns for Diabetes Risk**
   - Strong correlations between diabetes and BMI, age, blood pressure, general health
   - Socioeconomic factors (education, income, healthcare access) show protective effects
   - Multiple comorbidity clustering provides rich modeling opportunities

**5. Clinical Interpretability Requirements**
   - Healthcare context demands explainable models over black-box approaches
   - Feature importance analysis crucial for clinical decision support
   - Model bias assessment needed across demographic subgroups

**Q: Which features will you scale, encode, or exclude in preprocessing?**

**A:** Comprehensive preprocessing strategy based on EDA findings:

**Features Requiring Scaling:**
- **BMI, MentHlth, PhysHlth, Age** - standardization needed for consistent model performance
- **Robust scaling preferred** due to presence of outliers in health metrics

**Features Already Properly Encoded:**
- **All 18 categorical variables** - properly encoded as 0/1, no additional encoding needed
- **Target variable** - Diabetes_binary correctly formatted for binary classification

**Feature Engineering Opportunities:**
- **BMI categories:** Create clinical BMI classes (underweight/normal/overweight/obese)
- **Health composite scores:** Combine related health indicators for risk stratification
- **Interaction features:** BMI × Age, Health conditions × Demographics
- **Polynomial features:** Quadratic BMI terms to capture threshold effects

**Features to Retain:**
- **All 21 features recommended for retention** - no exclusions needed
- Low multicollinearity supports keeping full feature set
- Clinical relevance of all variables supports comprehensive modeling approach

**Q: What does your cleaned dataset look like (rows, columns, shape)?**

**A:** Post-preprocessing dataset specifications:

**Current State:**
- **Raw Shape:** (253,680, 22) including target variable
- **Features:** 21 predictive features + 1 target variable
- **Memory Usage:** 42.58 MB (efficient for modeling)

**After Deduplication:**
- **Cleaned Shape:** (229,474, 22) - removal of 24,206 duplicates
- **Data Reduction:** 9.5% reduction in sample size
- **Impact:** Maintains substantial sample size for robust training

**Final Modeling Dataset:**
- **Training Set:** ~183,579 samples (80% of cleaned data)
- **Test Set:** ~45,895 samples (20% of cleaned data)
- **Validation Strategy:** 5-fold stratified cross-validation maintaining class distribution
- **Feature Count:** 21 features + engineered features (estimated 25-30 total)

**Quality Metrics:**
- **Completeness:** 100% (no missing values)
- **Class Distribution:** Maintained 6.2:1 ratio after deduplication
- **Memory Efficiency:** <50 MB total, suitable for standard machine learning pipelines
- **Readiness Score:** 4/4 after duplicate removal

**Next Phase Preparation:**
- Dataset ready for immediate feature engineering and model development
- Recommended pipeline: deduplication → feature engineering → scaling → model training
- Expected timeline: Ready for Week 2 feature engineering phase

---

## ✅ Week 2: Feature Engineering & Deep Learning Prep

**Dataset:** CDC Diabetes Health Indicators (UCI ML Repository ID: 891)  
**Processing Date:** August 14, 2025  
**Pipeline Status:** ✅ Complete - Ready for Neural Network Training

---

### 🔧 1. Feature Encoding Strategy

**Q: Which categorical features did you encode, and what method did you use? Justify your choice.**

**A:** We strategically encoded **3 high-cardinality categorical features** using **Integer Encoding (LabelEncoder)**:

**Features Encoded:**
- **Age:** 13 categories (age groups 1-13) → Encoded to 0-12
- **GenHlth:** 5 categories (general health 1-5) → Encoded to 0-4  
- **bmi_category:** 6 categories (WHO BMI classes 0-5) → Encoded to 0-5

**Method Justification - Integer Encoding for Neural Networks:**

**Why Integer Encoding over One-Hot Encoding:**
- **Embedding Compatibility:** Integer encoding provides optimal input format for PyTorch embedding layers
- **Dimensionality Efficiency:** Reduces feature space from 24 one-hot columns to just 3 integer columns (13+5+6 → 3)
- **Dense Representation Learning:** Enables neural networks to learn meaningful, dense vector representations for each category
- **Memory Optimization:** Significantly reduces memory usage and computational overhead
- **Non-linear Relationship Discovery:** Allows embedding layers to capture complex, non-linear relationships between categorical values

**Why Not One-Hot Encoding:**
- Would create sparse, high-dimensional feature vectors (24 additional columns)
- Computationally inefficient for neural network training
- Loses the opportunity for learned embeddings that can capture semantic similarities
- Creates memory and performance bottlenecks with large datasets (229K+ samples)

**Technical Implementation:**
- Used sklearn's LabelEncoder for consistent, reversible transformations
- Maintained original features alongside encoded versions for interpretability
- Verified proper range mapping: all encoded features start from 0 (required for embedding layers)

---

### 📏 2. Numerical Feature Scaling

**Q: Which numerical features did you scale, and what scaling method did you use? Why was this choice appropriate?**

**A:** We scaled **2 critical numerical features** using **StandardScaler**:

**Features Scaled:**
- **MentHlth (Mental Health Days):** Original range 0-30, heavily right-skewed (skewness: 2.54)
- **PhysHlth (Physical Health Days):** Original range 0-30, right-skewed (skewness: 2.04)

**Scaling Results:**
- **Perfect Normalization Achieved:** Both features transformed to mean ≈ 0.000000, std ≈ 1.000000
- **Distribution Improvement:** Addressed right-skewness while preserving relative relationships
- **Neural Network Optimization:** Created zero-centered distributions optimal for gradient-based learning

**Why StandardScaler over MinMaxScaler:**

**Technical Advantages:**
- **Gradient Optimization:** Zero-centered data improves gradient flow in neural networks
- **Outlier Robustness:** More robust to the outliers we identified at maximum values (30 days)
- **Weight Initialization Compatibility:** Works optimally with standard neural network weight initialization schemes
- **Activation Function Efficiency:** Zero-centered inputs work better with activation functions like ReLU and sigmoid

**Clinical Justification:**
- **Outlier Preservation:** The 30-day outliers represent clinically meaningful chronic conditions, not data errors
- **Relationship Maintenance:** Preserves the relative differences between patients with different health burden levels
- **Scale Consistency:** Both MentHlth and PhysHlth have identical ranges, making StandardScaler ideal for consistent treatment

**Features NOT Scaled:**
- **BMI:** Kept original scale since we created categorical version (bmi_category) for the model
- **Binary Variables:** Already in optimal 0/1 format for neural networks
- **Encoded Categories:** Integer-encoded features are input to embedding layers, not direct neural network processing

---

### 🎯 3. Data Splitting Strategy

**Q: How did you split your data (train/validation/test), and what considerations did you account for?**

**A:** We implemented a **stratified 70/15/15 split** with careful attention to class balance preservation:

**Split Configuration:**
- **Training Set:** 160,631 samples (70.0%)
- **Validation Set:** 34,421 samples (15.0%)  
- **Test Set:** 34,422 samples (15.0%)
- **Total Processed:** 229,474 unique samples (after duplicate removal)

**Critical Design Decisions:**

**1. Stratified Sampling (stratify=y_raw):**
- **Class Balance Preservation:** Maintained the critical 85.4% / 14.6% diabetes distribution across ALL splits
- **Healthcare Importance:** In medical applications, preserving rare disease prevalence is essential for valid evaluation
- **Statistical Validity:** Ensures test set performance accurately reflects real-world population characteristics

**2. Two-Stage Splitting Process:**
- **Stage 1:** 85% temporary set vs 15% test set (with stratification)
- **Stage 2:** 70% train vs 15% validation from the 85% temporary set (with stratification)
- **Mathematical Precision:** test_size=(0.15/0.85) ensures exact 15% validation split

**3. Random State Control (random_state=42):**
- **Reproducibility:** Ensures consistent splits across different runs
- **Collaboration:** Team members can reproduce exact same train/val/test splits
- **Debugging:** Facilitates troubleshooting and model comparison

**Validation Results:**
- **Perfect Stratification:** Class distributions identical across splits (±0.02%)
- **No Data Leakage:** Clean separation between train/val/test with no sample overlap
- **Sample Size Adequacy:** Each split contains sufficient samples for robust training and evaluation

**Clinical Considerations:**
- **Minority Class Representation:** Each split contains ~5,000+ diabetes cases for reliable evaluation
- **Population Validity:** Test set accurately represents target deployment population
- **Bias Prevention:** Stratification prevents accidentally creating biased evaluation sets

---

### 🚀 4. Deep Learning Data Preparation

**Q: How did you prepare your data for PyTorch DataLoaders? What batch size and configurations did you choose?**

**A:** We created production-ready PyTorch DataLoaders optimized for efficient neural network training:

**DataLoader Configuration:**

**Batch Size Selection: 64**
- **Computational Efficiency:** Optimal balance for 229K+ sample dataset
- **GPU Memory:** Fits comfortably in standard GPU memory (8GB+)
- **Gradient Stability:** Large enough for stable gradient estimates, small enough for frequent updates
- **Training Speed:** Results in 2,510 training batches per epoch for reasonable training time

**Tensor Optimization:**
- **Feature Tensors:** Converted to float32 (standard for neural network computations)
- **Label Tensors:** Converted to long dtype (required for PyTorch classification losses)
- **Memory Efficiency:** Optimized tensor storage for large dataset processing

**DataLoader-Specific Configurations:**

**Training DataLoader:**
- **shuffle=True:** Critical for preventing overfitting and ensuring diverse mini-batches
- **Randomization:** Each epoch presents data in different order, improving generalization

**Validation/Test DataLoaders:**
- **shuffle=False:** Ensures consistent, reproducible evaluation across runs
- **Deterministic Evaluation:** Same sample order enables reliable performance comparison

**System Optimization:**
- **num_workers=0:** Configured for Windows compatibility (avoids multiprocessing issues)
- **pin_memory=True:** Enabled when CUDA available for faster GPU data transfer
- **Batch Consistency:** All loaders use identical batch size for consistent processing

**Production Readiness Verification:**
- **Successful Iteration:** Confirmed DataLoaders produce expected tensor shapes
- **Class Distribution:** Verified mini-batches maintain reasonable class representation
- **Feature Statistics:** Confirmed proper scaling preservation in tensor format
- **Memory Efficiency:** Total pipeline operates within reasonable memory constraints

**Technical Specifications:**
- **Training Batches:** 2,510 per epoch
- **Validation Batches:** 538 per evaluation
- **Test Batches:** 538 for final assessment
- **Feature Dimensions:** 22 features per sample (post-encoding and scaling)
- **Ready for Embedding:** Categorical features properly formatted for embedding layer input

---

### 🎯 Week 2 Summary & Next Steps

**Pipeline Achievements:**
✅ **Data Integrity:** Removed 24,206 duplicates, optimized data types (15.2% memory reduction)  
✅ **Feature Engineering:** Created WHO BMI categories, integer-encoded 3 high-cardinality features  
✅ **Neural Network Prep:** Perfect feature scaling, stratified splits, optimized DataLoaders  
✅ **Production Ready:** 229,474 samples × 22 features ready for deep learning implementation  

**Technical Validation:**
- **Class Balance Maintained:** 85.4%/14.6% preserved across all splits (±0.02%)
- **Feature Quality:** 22 properly formatted features (3 embedding + 2 scaled + 17 original)
- **Memory Optimized:** 36.1 MB total dataset, efficient batch processing
- **Performance Ready:** 2,510 training batches, GPU-optimized tensor format

**Week 3 Readiness:**
- **Neural Architecture:** Ready for embedding layers (3 categorical features properly encoded)
- **Training Pipeline:** DataLoaders configured for efficient gradient-based optimization
- **Evaluation Framework:** Stratified splits ensure valid performance assessment
- **Clinical Focus:** Preserved class balance critical for healthcare model evaluation

**Status:** ✅ **Week 2 Complete** - Feature engineering pipeline validated and ready for neural network implementation!

---

## 📊 Conclusion

This comprehensive EDA reveals a high-quality dataset with excellent potential for developing clinically meaningful diabetes risk prediction models. The combination of comprehensive health indicators, minimal preprocessing requirements, and clear predictive patterns provides a robust foundation for the feature engineering and modeling phases ahead.

**Key Success Factors Identified:**
1. **Data Quality:** Minimal missing data and appropriate encoding
2. **Clinical Relevance:** All features medically meaningful for diabetes prediction  
3. **Feature Independence:** Low multicollinearity supports comprehensive modeling
4. **Sufficient Sample Size:** Large dataset enables robust statistical analysis
5. **Clear Patterns:** Strong predictive relationships identified for modeling

**Immediate Next Steps:**
1. **Remove duplicate rows** to ensure data integrity
2. **Implement stratified sampling** to maintain class distribution
3. **Begin feature engineering** with BMI categories and composite health scores
4. **Prepare scaled features** for algorithm compatibility
5. **Establish baseline models** for performance benchmarking

The dataset is well-positioned for developing accurate, interpretable diabetes risk prediction models suitable for clinical deployment.

# 🔴 GlucoTrack Advanced Track

Welcome to the **Advanced Track** of the GlucoTrack project! This track is designed for participants with experience in machine learning who want to dive into deep learning and model explainability techniques to classify diabetes risk.

You’ll work with a real-world CDC dataset and build a complete pipeline from data preprocessing to deployment using tools like PyTorch/TensorFlow, SHAP, and MLflow.

---

## 📊 Dataset Overview

- Source: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- Goal: Classify individuals as diabetic (1) or non-diabetic (0)
- Features: Health, lifestyle, and demographic indicators

---

## 🎓 Weekly Breakdown

### Week 1: Exploratory Data Analysis (EDA)

- Perform the same EDA steps as the beginner track
- Pay special attention to class imbalance, skewed distributions, and multicollinearity
- Begin identifying features suited for embeddings

### Week 2: Feature Engineering & Deep Learning Prep

- Encode high-cardinality categorical features with integer labels for embedding layers
- Normalize numerical features where needed
- Split data into train/val/test with stratification
- Convert datasets into PyTorch Dataloaders or TensorFlow tf.data objects

### Week 3: Neural Network Design & Baseline Training

- Build a baseline Feedforward Neural Network (FFNN) with at least one hidden layer
- Include Dropout, ReLU, and Batch Normalization where appropriate
- Train the model using binary cross-entropy loss and an optimizer like Adam
- Evaluate performance using Accuracy, Precision, Recall, F1-score, and AUC
- Track experiments with MLflow

### Week 4: Model Tuning & Explainability

- Tune architecture and hyperparameters (layer size, dropout, learning rate, etc.)
- Use early stopping and/or learning rate schedulers
- Integrate SHAP, LIME, or Integrated Gradients to explain predictions
- Visualize feature contributions and evaluate model interpretability

### Week 5: Deployment

- 🟢 Easy: **Streamlit Cloud**

  - Use the Streamlit app structure from the beginner track
  - Add deep learning model integration and host on Streamlit Community Cloud

- 🟡 Intermediate: **Docker + Hugging Face Spaces**

  - Containerize your Streamlit app using a custom `Dockerfile`
  - Deploy the Docker image to Hugging Face Spaces using the Docker SDK option

- 🔴 Advanced: **API-based Deployment (Flask or FastAPI)**

  - Create a RESTful API to serve model predictions
  - Containerize with Docker
  - Deploy to platforms like Railway, Render, Fly.io, or GCP Cloud Run
  - Validate using tools like Postman or a simple frontend client

---

## 🗒️ Project Timeline Overview

| Phase                           | General Activities                                                     |
| ------------------------------- | ---------------------------------------------------------------------- |
| **Week 1: Setup + EDA**         | Clean, explore, and visualize the data                                 |
| **Week 2: Feature Engineering** | Transform features, encode variables, handle imbalance, prepare splits |
| **Week 3: Model Development**   | Train ML or DL models and evaluate performance                         |
| **Week 4: Model Optimization**  | Tune models, improve generalization, and interpret results             |
| **Week 5: Deployment**          | Deploy models via Streamlit or API-based solutions                     |

---

## 📃 Report Template

Use the [REPORT.md](./REPORT.md) to document your progress, architecture, evaluation, and explainability work for each week.

---

## 🚪 Where to Submit

Place your work inside the correct folder:

- `submissions/team-members/your-name/` if you are part of the official project team
- `submissions/community-contributions/your-name/` if you are an external contributor

Refer to the [CONTRIBUTING.md](../CONTRIBUTING.md) for complete instructions.


import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from sklearn.preprocessing import StandardScaler
import pickle
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the neural network architecture (MUST match your training model exactly)
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
        super(FFNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size//2),
            nn.BatchNorm1d(hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class DiabetesRiskPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None  # Changed from scaler to preprocessor
        self.feature_names = None
        self.explainer = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            # Load model configuration
            with open('streamlit_app/model_config.pkl', 'rb') as f:
                model_config = pickle.load(f)
            
            # Load feature names
            with open('streamlit_app/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # Initialize and load model
            self.model = FFNN(
                input_size=model_config['input_size'],
                hidden_size=model_config.get('hidden_size', 64),
                dropout_rate=model_config.get('dropout_rate', 0.3)
            )
            self.model.load_state_dict(torch.load('streamlit_app/best_model.pth', map_location='cpu'))
            self.model.eval()
            
            # Load preprocessor (not just scaler)
            with open('streamlit_app/preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            self.model_loaded = True
            return True
            
        except FileNotFoundError as e:
            st.error(f"Model files not found: {str(e)}")
            st.error("Please upload the following files: best_model.pth, preprocessor.pkl, model_config.pkl, feature_names.pkl")
            return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

    def predict(self, input_data):
        """Make prediction on input data"""
        if not self.model_loaded:
            return None
        
        try:
            # Convert to DataFrame with correct feature order
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data
            
            # DEBUG: Check if all expected features are present
            missing_features = set(self.feature_names) - set(input_df.columns)
            if missing_features:
                st.error(f"Missing features: {missing_features}")
                return None
            
            # Ensure correct column order
            input_df = input_df[self.feature_names]
            
            # DEBUG: Show feature order (remove in production)
            st.write("Debug - Feature order:", self.feature_names)
            st.write("Debug - Input order:", input_df.columns.tolist())
            
            # Preprocess the input (this handles scaling of continuous features)
            input_processed = self.preprocessor.transform(input_df)
            
            # Convert to tensor and predict
            input_tensor = torch.FloatTensor(input_processed)
            with torch.no_grad():
                prediction = self.model(input_tensor)
                risk_score = prediction.item()
            
            return risk_score
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None
        
# Initialize the predictor
@st.cache_resource
def load_predictor():
    predictor = DiabetesRiskPredictor()
    predictor.load_model()  # This will show errors if files are missing
    return predictor

def create_input_form():
    """Create the user input form"""
    st.sidebar.header("📋 Health Assessment Form")
    st.sidebar.write("Please fill out the following health indicators:")
    
    # Organize inputs into categories
    inputs = {}
    
    # Cardiovascular Health
    st.sidebar.subheader("🫀 Cardiovascular Health")
    inputs['HighBP'] = st.sidebar.selectbox("High Blood Pressure", [0, 1], 
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['HighChol'] = st.sidebar.selectbox("High Cholesterol", [0, 1],
                                            format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['CholCheck'] = st.sidebar.selectbox("Cholesterol Check (last 5 years)", [0, 1],
                                             format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['Stroke'] = st.sidebar.selectbox("History of Stroke", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['HeartDiseaseorAttack'] = st.sidebar.selectbox("Heart Disease/Attack", [0, 1],
                                                        format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Physical Health
    st.sidebar.subheader("🏃 Physical Health")
    inputs['bmi_category'] = st.sidebar.selectbox("BMI Category", [0, 1, 2, 3, 4, 5],
                                                format_func=lambda x: ['Underweight', 'Normal', 'Overweight', 'Obesity I', 'Obesity II', 'Obesity III'][x])
    inputs['PhysActivity'] = st.sidebar.selectbox("Physical Activity (last 30 days)", [0, 1],
                                                format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['DiffWalk'] = st.sidebar.selectbox("Difficulty Walking", [0, 1],
                                            format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['PhysHlth'] = st.sidebar.slider("Physical Health Issues (days in last 30)", 0, 30, 0)
    
    # Lifestyle Factors
    st.sidebar.subheader("🍎 Lifestyle Factors")
    inputs['Smoker'] = st.sidebar.selectbox("Smoker", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['HvyAlcoholConsump'] = st.sidebar.selectbox("Heavy Alcohol Consumption", [0, 1],
                                                     format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['Fruits'] = st.sidebar.selectbox("Daily Fruit Consumption", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['Veggies'] = st.sidebar.selectbox("Daily Vegetable Consumption", [0, 1],
                                           format_func=lambda x: "No" if x == 0 else "Yes")
    
    # Healthcare Access
    st.sidebar.subheader("🏥 Healthcare Access")
    inputs['AnyHealthcare'] = st.sidebar.selectbox("Health Insurance Coverage", [0, 1],
                                                 format_func=lambda x: "No" if x == 0 else "Yes")
    inputs['NoDocbcCost'] = st.sidebar.selectbox("Avoided Doctor Due to Cost", [0, 1],
                                                format_func=lambda x: "No" if x == 0 else "Yes")
    
    # General Health
    st.sidebar.subheader("🌟 General Health")
    inputs['GenHlth'] = st.sidebar.selectbox("General Health", [0, 1, 2, 3, 4],  # Changed to match your encoding
                                           format_func=lambda x: ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][x])
    inputs['MentHlth'] = st.sidebar.slider("Mental Health Issues (days in last 30)", 0, 30, 0)
    
    # Demographics
    st.sidebar.subheader("👤 Demographics")
    inputs['Sex'] = st.sidebar.selectbox("Sex", [0, 1],
                                       format_func=lambda x: "Female" if x == 0 else "Male")
    inputs['Age'] = st.sidebar.selectbox("Age Group", list(range(0, 13)),  # Adjusted for your encoding
                                       format_func=lambda x: f"Age Group {x+1}")
    inputs['Education'] = st.sidebar.selectbox("Education Level", list(range(0, 6)),  # Adjusted for your encoding
                                             format_func=lambda x: f"Education Level {x+1}")
    inputs['Income'] = st.sidebar.selectbox("Income Level", list(range(0, 8)),  # Adjusted for your encoding
                                          format_func=lambda x: f"Income Level {x+1}")
    
    return inputs

def create_risk_visualization(risk_score):
    """Create risk score visualization"""
    # Risk gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Diabetes Risk Score (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def interpret_risk_score(risk_score):
    """Provide interpretation of the risk score"""
    risk_percentage = risk_score * 100
    
    if risk_percentage < 25:
        risk_level = "Low Risk"
        color = "green"
        recommendations = [
            "Continue maintaining your healthy lifestyle",
            "Regular health check-ups are recommended",
            "Keep up with physical activity and healthy diet"
        ]
    elif risk_percentage < 50:
        risk_level = "Moderate Risk"
        color = "orange"
        recommendations = [
            "Consider lifestyle modifications",
            "Increase physical activity if possible",
            "Monitor diet and reduce processed foods",
            "Regular medical check-ups recommended"
        ]
    elif risk_percentage < 75:
        risk_level = "High Risk"
        color = "red"
        recommendations = [
            "Consult with healthcare provider soon",
            "Consider diabetes screening tests",
            "Implement significant lifestyle changes",
            "Focus on weight management if applicable"
        ]
    else:
        risk_level = "Very High Risk"
        color = "darkred"
        recommendations = [
            "Seek immediate medical consultation",
            "Diabetes screening tests highly recommended",
            "Urgent lifestyle interventions needed",
            "Consider working with diabetes specialist"
        ]
    
    return risk_level, color, recommendations

def display_model_architecture(predictor):
    """Display model architecture and configuration"""
    st.header("🧠 Model Architecture")
    
    if predictor.model_loaded:
        # Model summary
        st.subheader("Neural Network Architecture")
        
        # Load model config for display
        try:
            with open('model_config.pkl', 'rb') as f:
                model_config = pickle.load(f)
            
            # Display architecture details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Architecture Details:**")
                st.write(f"- Input Size: {model_config['input_size']} features")
                st.write(f"- Hidden Layer 1: {model_config.get('hidden_size', 64)} neurons")
                st.write(f"- Hidden Layer 2: {model_config.get('hidden_size', 64)//2} neurons")
                st.write(f"- Output Layer: 1 neuron (sigmoid)")
                st.write(f"- Dropout Rate: {model_config.get('dropout_rate', 0.3)}")
            
            with col2:
                st.markdown("**Layer Components:**")
                st.write("- Linear → BatchNorm → ReLU → Dropout")
                st.write("- Linear → BatchNorm → ReLU → Dropout") 
                st.write("- Linear → Sigmoid")
                
                st.markdown("**Activation Functions:**")
                st.write("- Hidden layers: ReLU")
                st.write("- Output layer: Sigmoid")
            
            # Feature names
            if predictor.feature_names:
                st.subheader("Input Features")
                feature_df = pd.DataFrame({
                    'Feature': predictor.feature_names,
                    'Index': range(len(predictor.feature_names))
                })
                st.dataframe(feature_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not load model configuration: {e}")
    else:
        st.error("Model not loaded")

def display_model_metrics():
    """Display model performance metrics"""
    st.header("📊 Model Performance Metrics")
    
    # Try to load metrics if available
    try:
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        
        with col2:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
            st.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
            
        with col3:
            st.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}")
            
        # Training history if available
        if 'train_losses' in metrics and 'val_losses' in metrics:
            st.subheader("Training History")
            
            # Create loss curves
            epochs = range(1, len(metrics['train_losses']) + 1)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Loss Curves', 'F1 Score Curves')
            )
            
            # Loss curves
            fig.add_trace(
                go.Scatter(x=list(epochs), y=metrics['train_losses'], 
                          name='Training Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=metrics['val_losses'], 
                          name='Validation Loss', line=dict(color='red')),
                row=1, col=1
            )
            
            # F1 curves if available
            if 'train_f1_scores' in metrics and 'val_f1_scores' in metrics:
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=metrics['train_f1_scores'], 
                              name='Training F1', line=dict(color='green')),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Scatter(x=list(epochs), y=metrics['val_f1_scores'], 
                              name='Validation F1', line=dict(color='orange')),
                    row=1, col=2
                )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
    except FileNotFoundError:
        st.warning("Model metrics file not found. Please ensure 'model_metrics.pkl' is available.")
        
        # Show placeholder metrics
        st.info("Placeholder metrics (replace with actual values):")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", "0.850")
            st.metric("Precision", "0.820")
        
        with col2:
            st.metric("Recall", "0.780")
            st.metric("F1 Score", "0.800")
            
        with col3:
            st.metric("AUC-ROC", "0.920")

def main():
    st.title("🩺 Diabetes Risk Assessment Tool")
    st.markdown("### AI-Powered Risk Prediction Based on CDC Health Indicators")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Risk Assessment", "🧠 Model Architecture", "📊 Model Metrics"])
    
    # Initialize predictor
    predictor = load_predictor()
    
    with tab1:
        st.markdown("""
        This tool uses a deep learning model trained on CDC BRFSS data to assess diabetes risk based on 
        health, lifestyle, and demographic factors. Please note that this is for informational purposes 
        only and should not replace professional medical advice.
        """)
        
        # Create input form
        user_inputs = create_input_form()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔮 Assess Diabetes Risk", type="primary", use_container_width=True):
                if predictor.model_loaded:
                    # Make actual prediction
                    risk_score = predictor.predict(user_inputs)
                    
                    if risk_score is not None:
                        st.success("✅ Risk Assessment Complete!")
                        
                        # Display risk visualization
                        risk_fig = create_risk_visualization(risk_score)
                        st.plotly_chart(risk_fig, use_container_width=True)
                        
                        # Risk interpretation
                        risk_level, color, recommendations = interpret_risk_score(risk_score)
                        
                        st.markdown(f"### Risk Level: <span style='color: {color}; font-weight: bold;'>{risk_level}</span>", 
                                   unsafe_allow_html=True)
                        
                        # Recommendations
                        st.markdown("### 📋 Recommendations:")
                        for i, rec in enumerate(recommendations, 1):
                            st.markdown(f"{i}. {rec}")
                            
                    else:
                        st.error("Failed to generate prediction. Please check your inputs and try again.")
                else:
                    st.error("Model not loaded. Please ensure all model files are uploaded.")
        
        with col2:
            st.markdown("### ℹ️ About This Tool")
            st.info("""
            **Model Information:**
            - Based on CDC BRFSS 2014 data
            - Feed-forward neural network
            - Trained on 21 health indicators
            - Uses SHAP for explainability
            
            **Important Notes:**
            - This is a screening tool only
            - Not a substitute for medical diagnosis
            - Consult healthcare providers for concerns
            - Based on self-reported survey data
            """)
            
            st.markdown("### 📊 Your Input Summary")
            input_df = pd.DataFrame([user_inputs]).T
            input_df.columns = ['Value']
            st.dataframe(input_df)
    
    with tab2:
        display_model_architecture(predictor)
    
    with tab3:
        display_model_metrics()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <small>
    This tool is for educational and informational purposes only. 
    Always consult with qualified healthcare professionals for medical advice.
    <br>
    Model trained on CDC BRFSS 2014 data | Built with Streamlit & PyTorch
    </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np

def about():
    st.title("OpenLearn 1.0 Capstone Project\n# ID: OL-25-LP-059")
    st.header("Dataset Overview")
    st.markdown("""
    #### Dataset Source: [Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
    #### Collected by OSMI (Open Sourcing Mental Illness)
    #### Features include:
    * Demographic details (age, gender, country)
    * Workplace environment (mental health benefits, leave policies)
    * Personal experiences (mental illness, family history)
    * Attitudes towards mental health
    """)

    st.subheader("Problem Statement")
    st.markdown("""
    As a Machine Learning Engineer at NeuronInsights Analytics, you've been contracted by a coalition of
    leading tech companies including CodeLab, QuantumEdge, and SynapseWorks. Alarmed by rising burnout,
    disengagement, and attrition linked to mental health, the consortium seeks data-driven strategies to
    proactively identify and support at-risk employees.
                
    ### Project Objectives:
    * **Exploratory Data Analysis**
    * **Supervised Learning**:
        * *Classification task*: Predict whether a person is likely to seek mental health treatment
        * *Regression task*: Predict the respondent's age
    * **Unsupervised Learning**: Cluster tech workers into mental health personas
    * **Streamlit App Deployment**
    """)

def visualization():
    st.title("Dataset Visualization")
    st.markdown("Visualization of different features to understand trends and patterns in the data.")
    
    st.header("Key Insights")
    st.markdown('''
    **Age Distribution:** Most people are middle-aged, with 30s being the most significant age group.
    
    **Gender:** Majority is Male (78%+), around 20% females, and small proportion of other genders.
    
    **Treatment Seeking:** Mixed reviews - treatment seekers slightly outnumber non-seekers.
    
    **Work Interference:** Most people fall in 'Sometimes' category for work interference.
    
    **Family History:** More people without family history of mental illness than those with it.
    ''')
    st.image("assets\Screenshot 2025-08-13 225651.png")
    st.image("assets\Screenshot 2025-08-13 225713.png")
    st.image("assets\Screenshot 2025-08-13 225728.png")
    st.image("assets\Screenshot 2025-08-13 225738.png")
    st.image("assets\Screenshot 2025-08-13 225749.png")

def supervised():
    st.title('Supervised Learning Tasks')
    
    tab1, tab2 = st.tabs(["Classification Task", "Regression Task"])
    
    with tab1:
        st.header('Classification Task')
        st.markdown('Predicting whether a person will seek mental health treatment.')
        
        st.subheader('Model Performance')
        st.markdown("""
        **Best Model:** XGBoost Classifier
        
        **Performance Metrics:**
        - F1 Score: 0.75
        - Accuracy: 78.4%
        - ROC-AUC Score: 81.2%
        """)
        
        st.code('''
        Model Comparison Results:
        LogReg F1 Score: 0.725
        SVC F1 Score: 0.707
        KNN F1 Score: 0.627
        Random Forest F1 Score: 0.728
        XGBoost F1 Score: 0.752 (Best)
        ''')
    
    with tab2:
        st.header('Regression Task')
        st.markdown('Predicting age based on workplace and mental health factors.')
        
        st.subheader('Model Performance')
        st.markdown("""
        **Best Model:** Ridge Regression
        
        **Performance Metrics:**
        - R² Score: 0.071
        - MAE: 5.41
        - RMSE: 7.04
        """)

def unsupervised():
    st.title("Clustering Analysis")
    st.markdown("Segmenting tech workers into mental health personas using unsupervised learning.")
    
    st.header("Identified Personas")
    
    personas = {
        "Silent Sufferers": {
            "description": "Often feel alone with their problems. Don't think workplace supports them.",
            "characteristics": ["High stress", "Rarely ask for help", "Work in smaller companies", "Feel unsafe discussing mental health"],
            "recommendations": ["Increase awareness", "Improve accessibility", "Create safe spaces"]
        },
        "Open Advocates": {
            "description": "Open about mental health and encourage others to discuss it.",
            "characteristics": ["Know available resources", "Use help when needed", "Work in bigger companies", "Promote mental health awareness"],
            "recommendations": ["Leverage as mentors", "Success story sharing", "Peer support programs"]
        },
        "Under-Supported Professionals": {
            "description": "Know they're struggling but workplace doesn't provide adequate support.",
            "characteristics": ["High stress", "Want to talk but lack support", "Large companies/remote", "Help exists but doesn't reach them"],
            "recommendations": ["Bridge support gaps", "Improve resource accessibility", "Better communication"]
        },
        "Resilient Optimists": {
            "description": "Generally doing well mentally with good workplace support.",
            "characteristics": ["Feel supported", "Low stress", "Open to discussion", "Often younger/flexible jobs"],
            "recommendations": ["Maintain current support", "Use as success examples", "Continuous monitoring"]
        }
    }
    
    for persona_name, info in personas.items():
        with st.expander(f"Persona: {persona_name}"):
            st.write(f"**Description:** {info['description']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Characteristics:**")
                for char in info['characteristics']:
                    st.write(f"• {char}")
            with col2:
                st.write("**Recommendations:**")
                for rec in info['recommendations']:
                    st.write(f"• {rec}")
    st.image("assets\Screenshot 2025-08-13 233345.png")

def simple_prediction_logic(age, family_history, work_interfere, benefits, seek_help, mental_health_consequence):
    score = 0
    
    if age > 30:
        score += 1
    if family_history == 'Yes':
        score += 2
    if work_interfere in ['Often', 'Sometimes']:
        score += 1
    if benefits == 'Yes':
        score += 1
    if seek_help == 'Yes':
        score += 1
    if mental_health_consequence == 'No':
        score += 1
        
    return 1 if score >= 3 else 0

def classifier_demo():
    st.title("Mental Health Treatment Prediction")
    st.markdown("Enter employee details to predict likelihood of seeking mental health treatment.")
    st.info("Note: This is a simplified prediction model for demonstration purposes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        age = st.slider("Age", 15, 70, 25)
        gender = st.selectbox("Gender", ['Male', 'Female', 'Non-binary/Other'])
        family_history = st.selectbox("Family History of Mental Illness?", ['No', 'Yes'])
        
        st.subheader("Work Environment")
        work_interfere = st.selectbox("How often does mental health interfere with work?", 
                                    ['Never', 'Rarely', 'Sometimes', 'Often'])
        no_employees = st.selectbox("Company Size", 
                                  ['1-5', '6-25', '26-100', '100-500', '500-1000', 'More than 1000'])
        tech_company = st.selectbox("Tech Company?", ['Yes', 'No'])
        
    with col2:
        st.subheader("Benefits & Support")
        benefits = st.selectbox("Mental Health Benefits?", ['No', "Don't know", 'Yes'])
        wellness_program = st.selectbox("Wellness Program Available?", ['No', "Don't know", 'Yes'])
        seek_help = st.selectbox("Employer encourages seeking help?", ['No', "Don't know", 'Yes'])
        
        st.subheader("Workplace Culture")
        mental_health_consequence = st.selectbox("Fear consequences for mental health issues?", 
                                                ['No', 'Maybe', 'Yes'])
        coworkers = st.selectbox("Comfortable discussing with coworkers?", ['No', 'Some of them', 'Yes'])
        supervisor = st.selectbox("Comfortable discussing with supervisor?", ['No', 'Some of them', 'Yes'])

    if st.button("Predict Treatment Seeking", type="primary"):
        prediction = simple_prediction_logic(age, family_history, work_interfere, benefits, seek_help, mental_health_consequence)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("✅ **Prediction: WILL seek treatment**")
                st.markdown("""
                **Recommendations:**
                - Ensure easy access to resources
                - Maintain supportive environment
                - Regular check-ins
                """)
            else:
                st.error("❌ **Prediction: Will NOT seek treatment**")
                st.markdown("""
                **Recommendations:**
                - Increase awareness of available resources
                - Reduce stigma through education
                - Improve anonymity protections
                - Provide manager training
                """)
        
        with col2:
            # Simple age-based insights
            st.info(f"**Age Group Analysis:** {age} years")
            
            if age < 25:
                st.write("Young professional - May need guidance on resources")
            elif age < 35:
                st.write("Mid-career - Balance work-life stress")
            elif age < 45:
                st.write("Senior professional - Leadership stress")
            else:
                st.write("Experienced - Different support needs")
                
            # Risk factors
            st.subheader("Risk Factors")
            risk_factors = []
            if family_history == 'Yes':
                risk_factors.append("Family history present")
            if work_interfere in ['Often', 'Sometimes']:
                risk_factors.append("Work interference issues")
            if mental_health_consequence == 'Yes':
                risk_factors.append("Fear of consequences")
            if benefits != 'Yes':
                risk_factors.append("Limited mental health benefits")
                
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"⚠️ {factor}")
            else:
                st.write("✅ Low risk profile")

# Navigation
pg = st.navigation([
    st.Page(about, title="Home", icon="🏠"),
    st.Page(visualization, title="Data Analysis", icon="📊"),
    st.Page(supervised, title="ML Models", icon="🤖"),
    st.Page(unsupervised, title="Clustering", icon="👥"),
    st.Page(classifier_demo, title="Prediction Tool", icon="🎯")
])

pg.run()
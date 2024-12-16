###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('CatBoost frailty.pkl')
scaler = joblib.load('scaler frailty.pkl') 

# Define feature options
cognitive_state_options = {    
    0: ' Normal(0)',    
    1: 'Mild(1)',    
    2: 'Moderate(2)',    
    3: ' Severe(3)',
}

vegetable_options = {       
    0: '＜300(0)',    
    1: '300-500(1)',    
    2: ' ＞350(2)',
}

education_options = {       
    0: 'Below junior high school(0)',    
    1: 'Senior high school(1)',    
    2: ' College or above(2)',
}




# Define feature names
feature_names = ["cognitive state, "vegetable", "age", "medicine", "MCC", "HGB", "NLR","TC","education"]

# Streamlit user interface
st.title("Breast Cancer Recurrence Predictor")


# cognitive_state
cognitive_state = st.selectbox("cognitive_state:", options=list(cognitive_state.keys()), format_func=lambda x: cognitive_state_options[x])

# vegetable
vegetable = st.selectbox("vegetable_state:", options=list(vegetable_state.keys()), format_func=lambda x: vegetable_state_options[x])

# age
age = st.number_input("age:", min_value=0, max_value=120, value=65)

# medicine
medicine = st.selectbox("medicine:", options=[0, 1], format_func=lambda x: '＜5' if x == 0 else '≥5')

# MCC
MCC= st.number_input("MCC:", min_value=0, max_value=10, value=0)

# HGB
HGB = st.number_input("HGB:", min_value=0, max_value=200, value=100)

# NLR
NLR = st.number_input("NLR:", min_value=0, max_value=20, value=5)

# TC
age = st.number_input("age:", min_value=0, max_value=20, value=5)

# education
education = st.selectbox("education:", options=list(education.keys()), format_func=lambda x: education_options[x])




# 准备输入特征
feature_values = [cognitive state, vegetable, age, medicine, MCC, HGB, NLR,TC,education]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [age, MCC,HGB,NLR,TC]
categorical_features=[cognitive state, vegetable,medicine,education]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)


# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=["age","MCC","HGB","NLR","TC"])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)


# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)


if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(0: No Disease,1: Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:        
         advice = (            
                f"According to our model, you have a high risk of frailty. "            
                f"The model predicts that your probability of having frailty is {probability:.1f}%. "            
                "It's advised to consult with your healthcare provider for further evaluation and possible intervention."        
          )    
    else:        
         advice = (           
                f"According to our model, you have a low risk of frailty. "            
                f"The model predicts that your probability of not having frailty is {probability:.1f}%. "            
                "However, maintaining a healthy lifestyle is important. Please continue regular check-ups with your healthcare provider."        
          )    
    st.write(advice)

    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")

  # 创建SHAP解释器
    explainer_shap = shap.TreeExplainer(model)
    
    # 获取SHAP值
    shap_values = explainer_shap.shap_values(pd.DataFrame(final_features_df,columns=feature_names))
    
  # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

# Display the SHAP force plot for the predicted class    
    if predicted_class == 1:        
        shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], original_feature_values, matplotlib=True)    
    else:        
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], original_feature_values, matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)    
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')

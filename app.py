

# import streamlit as st
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# import numpy as np
# import tensorflow as tf

# # Initialize session state for navigation
# if 'submitted' not in st.session_state:
#     st.session_state.submitted = False
# if 'severity' not in st.session_state:
#     st.session_state.severity = None
# # Load your dataset (ensure to update the file path)
# file_path = 'expanded_synthetic_health_data.csv'
# df = pd.read_csv(file_path)

# # Initialize encoders and scaler
# scaler = StandardScaler()
# label_encoder = LabelEncoder()

# # Define mappings (same as in your model code)
# gender_mapping = {'Male': 1, 'Female': 0}
# symptom_mapping = {
#     "Abdominal pain": 1.0, "Chest pain": 2.0, "Constipation": 3.0, "Cough": 4.0, "Diarrhea": 5.0,
#     "Difficulty swallowing": 6.0, "Dizziness": 7.0, "Eye discomfort and redness": 8.0,
#     "Foot pain or ankle pain": 9.0, "Foot swelling or leg swelling": 10.0, "Headaches": 11.0,
#     "Heart palpitations": 12.0, "Hip pain": 13.0, "Knee pain": 14.0, "Low back pain": 15.0,
#     "Nasal congestion": 16.0, "Nausea or vomiting": 17.0, "Neck pain": 18.0, "Numbness or tingling in hands": 19.0,
#     "Shortness of breath": 20.0, "Shoulder pain": 21.0, "Sore throat": 22.0, "Urinary problems": 23.0,
#     "Wheezing": 24.0, "Ear ache": 25.0, "Fever": 26.0, "Joint pain or muscle pain": 27.0, "Skin rashes": 28.0
# }
# symptom_duration_mapping = {'Less than 2 days': 0, '2-5 days': 1, 'More than 5 days': 2}
# onset_mapping = {'Sudden': 1, 'Gradual': 0}
# chronic_mapping = {
#     "Diabetes": 1.0, "Hypertension": 2.0, "Asthma": 3.0, "Arthritis": 4.0, "Obesity": 5.0,
#     "Cholesterol": 6.0, "Depression": 7.0, "Cirrhosis": 8.0, "No chronic conditions": 9.0
# }
# alcohol_mapping = {'No': 0, 'Occasionally': 1, 'Regularly': 2}
# physical_mapping = {'No': 0, 'Light': 1, 'Moderate': 2, 'Intense': 3}
# sleep_mapping = {'Excellent': 3.0, 'Good': 2.0, 'Fair': 1.0, 'Poor': 0.0}

# symptoms_medications = {
#     "Abdominal pain": "Antacids, Antispasmodics, Proton Pump Inhibitors, Analgesics",
#     "Chest pain": "Nitroglycerin, Aspirin, Proton Pump Inhibitors, Muscle relaxants",
#     "Constipation": "Laxatives, Stool softeners, Fiber supplements",
#     "Cough": "Cough suppressants, Expectorants, Antihistamines, Bronchodilators",
#     "Diarrhea": "Antidiarrheal agents, Oral rehydration solutions, Probiotics",
#     "Difficulty swallowing": "Proton Pump Inhibitors, Antacids",
#     "Dizziness": "Antivertigo agents, Benzodiazepines, Hydration and electrolytes",
#     "Eye discomfort and redness": "Artificial tears, Antihistamine eye drops, Antibiotic eye drops",
#     "Foot pain or ankle pain": "NSAIDs, Topical analgesics",
#     "Foot swelling or leg swelling": "Diuretics, Compression stockings",
#     "Headaches": "NSAIDs, Acetaminophen, Triptans, Caffeine-containing medications",
#     "Heart palpitations": "Beta-blockers, Calcium channel blockers, Antiarrhythmic drugs",
#     "Hip pain": "NSAIDs, Corticosteroid injections",
#     "Knee pain": "NSAIDs, Topical analgesics, Corticosteroid injections",
#     "Low back pain": "NSAIDs, Muscle relaxants, Topical pain relievers",
#     "Nasal congestion": "Decongestants, Nasal sprays",
#     "Nausea or vomiting": "Antiemetics, Antacids, Ginger supplements",
#     "Neck pain": "NSAIDs, Muscle relaxants",
#     "Numbness or tingling in hands": "NSAIDs, Gabapentin, Vitamin B12 supplements",
#     "Shortness of breath": "Bronchodilators, Inhaled corticosteroids, Diuretics",
#     "Shoulder pain": "NSAIDs, Topical analgesics",
#     "Sore throat": "Throat lozenges, NSAIDs, Saltwater gargle",
#     "Urinary problems": "Antibiotics, Alpha-blockers",
#     "Wheezing": "Bronchodilators, Inhaled corticosteroids, Leukotriene inhibitors",
#     "Ear ache": "Analgesics, Antibiotic ear drops",
#     "Fever": "Antipyretics, Hydration",
#     "Joint pain or muscle pain": "NSAIDs, Topical analgesics, Glucosamine supplements, Corticosteroid injections",
#     "Skin rashes": "Antihistamines, Topical corticosteroids, Antibiotic creams"
# }

# # Prepare data for model
# df['Gender'] = df['Gender'].map(gender_mapping)
# df['General Symptoms'] = df['General Symptoms'].map(symptom_mapping)
# df['Pain Scale'] = df['Pain Scale'] / 10.0
# df['Symptom Duration'] = label_encoder.fit_transform(df['Symptom Duration'])
# df['Onset'] = df['Onset'].map(onset_mapping)
# df['Chronic Conditions'] = label_encoder.fit_transform(df['Chronic Conditions'].fillna(''))
# df['Allergies'] = df['Allergies'].map({'Yes': 1, 'No': 0})
# df['Medications'] = df['Medications'].map({'Yes': 1, 'No': 0})
# df['Travel History'] = df['Travel History'].map({'Yes': 1, 'No': 0})
# df['Contact with Sick Individuals'] = df['Contact with Sick Individuals'].map({'Yes': 1, 'No': 0})
# df['Smoking'] = df['Smoking'].map({'Yes': 1, 'No': 0})
# df['Alcohol Consumption'] = df['Alcohol Consumption'].map(alcohol_mapping)
# df['Physical Activity'] = df['Physical Activity'].map(physical_mapping)
# df['Stress Levels'] = df['Stress Levels'] / 10.0
# df['Sleep Quality'] = df['Sleep Quality'].map(sleep_mapping)

# # Normalize numeric columns
# numeric_columns = ['Age', 'Symptom Duration', 'Chronic Conditions', 'Alcohol Consumption', 'Physical Activity', 'Sleep Quality']
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# # Encode target variable
# df['Severity'] = label_encoder.fit_transform(df['Severity'])

# # Split dataset into features (X) and target (y)
# X = df.drop(columns=['Severity'])
# y = df['Severity'].values

# # Load or build the neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(X.shape[1],)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
# ])

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# def predict_severity(user_data):
#     user_df = pd.DataFrame([user_data], columns=X.columns)
#     user_df['Gender'] = user_df['Gender'].map(gender_mapping)
#     user_df['General Symptoms'] = user_df['General Symptoms'].map(symptom_mapping)
#     user_df['Symptom Duration'] = user_df['Symptom Duration'].map(symptom_duration_mapping)
#     user_df['Onset'] = user_df['Onset'].map(onset_mapping)
#     chronic_conditions = user_df['Chronic Conditions'].values[0].split(', ')
#     user_df['Chronic Conditions'] = len(chronic_conditions)
#     user_df['Allergies'] = 1 if user_df['Allergies'].values[0] == 'Yes' else 0
#     user_df['Medications'] = 1 if user_df['Medications'].values[0] == 'Yes' else 0
#     user_df['Travel History'] = 1 if user_df['Travel History'].values[0] == 'Yes' else 0
#     user_df['Contact with Sick Individuals'] = 1 if user_df['Contact with Sick Individuals'].values[0] == 'Yes' else 0
#     user_df['Smoking'] = 1 if user_df['Smoking'].values[0] == 'Yes' else 0
#     user_df['Alcohol Consumption'] = alcohol_mapping[user_df['Alcohol Consumption'].values[0]]
#     user_df['Physical Activity'] = physical_mapping[user_df['Physical Activity'].values[0]]
#     user_df['Stress Levels'] = user_df['Stress Levels'] / 10.0
#     user_df['Sleep Quality'] = sleep_mapping[user_df['Sleep Quality'].values[0]]
#     user_df[numeric_columns] = scaler.transform(user_df[numeric_columns])
#     severity_prediction = model.predict(user_df)
#     severity_class = label_encoder.inverse_transform([np.argmax(severity_prediction)])
#     return severity_class[0]

# st.title("MEDICAL SURVEY PAGE")

# if not st.session_state.get('submitted', False):
#     st.write("#### Please fill out the following questionnaire to assess severity:")

#     # Input fields
#     gender = st.selectbox("Gender", ['Male', 'Female'])
#     age = st.number_input("Age", min_value=0, value=30)
#     pain_scale = st.slider("Rate the  Pain you are facing from 0 to 10", 0, 10, 5)
#     general_symptom = st.selectbox("Mention the General Symptom you are facing", list(symptom_mapping.keys()))
#     symptom_duration = st.selectbox("What is the Duration of the Symptom", ['Less than 2 days', '2-5 days', 'More than 5 days'])
#     onset = st.selectbox("What is the Onset of the symptom you face currently", ['Sudden', 'Gradual'])
#     chronic_conditions = st.selectbox("Chronic Conditions you aldready have", ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Obesity', 'Cholesterol', 'Depression', 'Cirrhosis', 'No chronic conditions'])
#     allergies = st.selectbox("Do you have allergies?", ['Yes', 'No'])
#     medications = st.selectbox("Are you currently taking medications?", ['Yes', 'No'])
#     travel_history = st.selectbox("Did you recently travel anywhere?", ['Yes', 'No'])
#     contact_sick = st.selectbox("Did you have Contact with sick individuals?", ['Yes', 'No'])
#     smoking = st.selectbox("Do you smoke?", ['Yes', 'No'])
#     alcohol_consumption = st.selectbox("Alcohol Consumption", ['No', 'Occasionally', 'Regularly'])
#     physical_activity = st.selectbox("Physical Activity", ['No', 'Light', 'Moderate', 'Intense'])
#     stress_levels = st.slider("select your Stress Level (0 to 10)", 0, 10, 5)
#     sleep_quality = st.selectbox("How is your Sleep Quality?", ['Excellent', 'Good', 'Fair', 'Poor'])

#     user_data = {
#         'Gender': gender,
#         'Age': age,
#         'Pain Scale': pain_scale,
#         'General Symptoms': general_symptom,
#         'Symptom Duration': symptom_duration,
#         'Onset': onset,
#         'Chronic Conditions': chronic_conditions,
#         'Allergies': allergies,
#         'Medications': medications,
#         'Travel History': travel_history,
#         'Contact with Sick Individuals': contact_sick,
#         'Smoking': smoking,
#         'Alcohol Consumption': alcohol_consumption,
#         'Physical Activity': physical_activity,
#         'Stress Levels': stress_levels,
#         'Sleep Quality': sleep_quality
#     }

#     # Submit button
#     if st.button("Submit"):
#         # Save the user data and mark submission
#         st.session_state.submitted = True
#         st.session_state.user_data = user_data
#         st.session_state.severity = predict_severity(user_data)

# else:
#     # Display predicted severity in a large bold box
#     severity = st.session_state.severity
#     st.write("### ")
#     st.write("### Severity Prediction Result")
#     st.markdown(f"""<div style='font-size: 32px; font-weight: bold; padding: 20px; text-align: center; border: 2px solid #209cee; border-radius: 10px;'>Predicted Severity Level<br><div style='font-size: 40px; font-weight: bold; color: #209cee;;'>{severity}</div></div>""",unsafe_allow_html=True)
#     st.write("### ")
#     def show_prescription(severity):
#         if severity == 'Mild':
#             st.write("### RECOMENDATIONS:")
#             st.write("- Rest without any disturbance.")
#             st.write("- Drink lots of water.")
#             st.write("- Don't stress too much .It just mild case")
#             st.write("- Try to avoid processed food for brief time.")
#         elif severity == 'Moderate':
#             st.write("### RECOMENDATIONS:")
#             st.write("- Rest without any disturbance.")
#             st.write("- Drink lots of water.")
#             st.write("- Don't stress too much.")
#             st.write("- Avoid alcohol and cigarettes if the habit persists")
#             st.write("- Don't sit for too long.try to have small walk")
#         elif severity == 'Severe':
#             st.write("### RECOMENDATIONS:")
#             st.write("- Do not take any medication on your own to counter the symptoms")
#             st.write("- Continue with your existing medication if any ")
#             st.write("- try to get Rest and dont get streesed")
#             st.write("- Book an appointment with our doctor by clicking ""Doctor appointment"" for quick services")
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     show_prescription(severity)
#     # Show recommended medications
    
#     selected_symptom = st.session_state.user_data['General Symptoms']
#     medications_list = symptoms_medications.get(selected_symptom, "No specific medication found")
#     #st.write(f"Recommended Medications for {selected_symptom}: **{medications_list}**")

    
        
#     # Back button to return to the previous page
#     if st.button("Back"):
#         st.session_state.submitted = False
#         st.session_state.severity = None

#     # Create two columns for buttons
#     col1, col2 = st.columns(2)

#     with col1:
#         if st.button("Doctor Appointment"):
#             # Redirect to a new page with summary and prescription
#             st.session_state.appointment = True
#             st.session_state.severity = severity
#             st.session_state.meds_list = medications_list
#             st.experimental_rerun()

#     with col2:
#         if st.button("Order Medicine"):
#             # Redirect to a new page with invoice-like format
#             st.session_state.order = True
#             st.session_state.meds_list = medications_list
#             st.experimental_rerun()

#     # Check if the appointment or order button was pressed
#     if st.session_state.get('appointment', False):
#         st.session_state.appointment = False
        
#         st.write("### Patient Information Summary for appointment confirmation:")
#         st.write("###### Check your given details before confirming your appointment with the Doctor")
#         for key, value in st.session_state.user_data.items():
#             st.markdown(f"<div style='display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid #ddd;'><span style='font-weight:bold;'>{key}:</span><span>{value}</span></div>", unsafe_allow_html=True)
#         st.write("### ")
#         st.button("Book the appointment")
        

#     if st.session_state.get('order', False):
#         st.session_state.order = False
#         medications_invoice = pd.DataFrame({
#             "Medication": medications_list.split(', '),
#             "Nos": [1] * len(medications_list.split(', '))  # Example fixed price per medication
#         })
#         #medications_invoice['Total'] = medications_invoice['Quantity'] * medications_invoice['Price']
#         st.write("### Invoice for Recommended Medications:")
#         st.table(medications_invoice)
#         st.button("Confirm Order")



import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import tensorflow as tf

# Initialize session state for navigation
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'severity' not in st.session_state:
    st.session_state.severity = None
# Load your dataset (ensure to update the file path)
file_path = 'expanded_synthetic_health_data.csv'
df = pd.read_csv(file_path)

# Initialize encoders and scaler
scaler = StandardScaler()
label_encoder = LabelEncoder()

# Define mappings (same as in your model code)
gender_mapping = {'Male': 1, 'Female': 0}
symptom_mapping = {
    "Abdominal pain": 1.0, "Chest pain": 2.0, "Constipation": 3.0, "Cough": 4.0, "Diarrhea": 5.0,
    "Difficulty swallowing": 6.0, "Dizziness": 7.0, "Eye discomfort and redness": 8.0,
    "Foot pain or ankle pain": 9.0, "Foot swelling or leg swelling": 10.0, "Headaches": 11.0,
    "Heart palpitations": 12.0, "Hip pain": 13.0, "Knee pain": 14.0, "Low back pain": 15.0,
    "Nasal congestion": 16.0, "Nausea or vomiting": 17.0, "Neck pain": 18.0, "Numbness or tingling in hands": 19.0,
    "Shortness of breath": 20.0, "Shoulder pain": 21.0, "Sore throat": 22.0, "Urinary problems": 23.0,
    "Wheezing": 24.0, "Ear ache": 25.0, "Fever": 26.0, "Joint pain or muscle pain": 27.0, "Skin rashes": 28.0
}
symptom_duration_mapping = {'Less than 2 days': 0, '2-5 days': 1, 'More than 5 days': 2}
onset_mapping = {'Sudden': 1, 'Gradual': 0}
chronic_mapping = {
    "Diabetes": 1.0, "Hypertension": 2.0, "Asthma": 3.0, "Arthritis": 4.0, "Obesity": 5.0,
    "Cholesterol": 6.0, "Depression": 7.0, "Cirrhosis": 8.0, "No chronic conditions": 9.0
}
alcohol_mapping = {'No': 0, 'Occasionally': 1, 'Regularly': 2}
physical_mapping = {'No': 0, 'Light': 1, 'Moderate': 2, 'Intense': 3}
sleep_mapping = {'Excellent': 3.0, 'Good': 2.0, 'Fair': 1.0, 'Poor': 0.0}

symptoms_medications = {
    "Abdominal pain": "Antacids, Antispasmodics, Proton Pump Inhibitors, Analgesics",
    "Chest pain": "Nitroglycerin, Aspirin, Proton Pump Inhibitors, Muscle relaxants",
    "Constipation": "Laxatives, Stool softeners, Fiber supplements",
    "Cough": "Cough suppressants, Expectorants, Antihistamines, Bronchodilators",
    "Diarrhea": "Antidiarrheal agents, Oral rehydration solutions, Probiotics",
    "Difficulty swallowing": "Proton Pump Inhibitors, Antacids",
    "Dizziness": "Antivertigo agents, Benzodiazepines, Hydration and electrolytes",
    "Eye discomfort and redness": "Artificial tears, Antihistamine eye drops, Antibiotic eye drops",
    "Foot pain or ankle pain": "NSAIDs, Topical analgesics",
    "Foot swelling or leg swelling": "Diuretics, Compression stockings",
    "Headaches": "NSAIDs, Acetaminophen, Triptans, Caffeine-containing medications",
    "Heart palpitations": "Beta-blockers, Calcium channel blockers, Antiarrhythmic drugs",
    "Hip pain": "NSAIDs, Corticosteroid injections",
    "Knee pain": "NSAIDs, Topical analgesics, Corticosteroid injections",
    "Low back pain": "NSAIDs, Muscle relaxants, Topical pain relievers",
    "Nasal congestion": "Decongestants, Nasal sprays",
    "Nausea or vomiting": "Antiemetics, Antacids, Ginger supplements",
    "Neck pain": "NSAIDs, Muscle relaxants",
    "Numbness or tingling in hands": "NSAIDs, Gabapentin, Vitamin B12 supplements",
    "Shortness of breath": "Bronchodilators, Inhaled corticosteroids, Diuretics",
    "Shoulder pain": "NSAIDs, Topical analgesics",
    "Sore throat": "Throat lozenges, NSAIDs, Saltwater gargle",
    "Urinary problems": "Antibiotics, Alpha-blockers",
    "Wheezing": "Bronchodilators, Inhaled corticosteroids, Leukotriene inhibitors",
    "Ear ache": "Analgesics, Antibiotic ear drops",
    "Fever": "Antipyretics, Hydration",
    "Joint pain or muscle pain": "NSAIDs, Topical analgesics, Glucosamine supplements, Corticosteroid injections",
    "Skin rashes": "Antihistamines, Topical corticosteroids, Antibiotic creams"
}

# Prepare data for model
df['Gender'] = df['Gender'].map(gender_mapping)
df['General Symptoms'] = df['General Symptoms'].map(symptom_mapping)
df['Pain Scale'] = df['Pain Scale'] / 10.0
df['Symptom Duration'] = label_encoder.fit_transform(df['Symptom Duration'])
df['Onset'] = df['Onset'].map(onset_mapping)
df['Chronic Conditions'] = label_encoder.fit_transform(df['Chronic Conditions'].fillna(''))
df['Allergies'] = df['Allergies'].map({'Yes': 1, 'No': 0})
df['Medications'] = df['Medications'].map({'Yes': 1, 'No': 0})
df['Travel History'] = df['Travel History'].map({'Yes': 1, 'No': 0})
df['Contact with Sick Individuals'] = df['Contact with Sick Individuals'].map({'Yes': 1, 'No': 0})
df['Smoking'] = df['Smoking'].map({'Yes': 1, 'No': 0})
df['Alcohol Consumption'] = df['Alcohol Consumption'].map(alcohol_mapping)
df['Physical Activity'] = df['Physical Activity'].map(physical_mapping)
df['Stress Levels'] = df['Stress Levels'] / 10.0
df['Sleep Quality'] = df['Sleep Quality'].map(sleep_mapping)

# Normalize numeric columns
numeric_columns = ['Age', 'Symptom Duration', 'Chronic Conditions', 'Alcohol Consumption', 'Physical Activity', 'Sleep Quality']
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Encode target variable
df['Severity'] = label_encoder.fit_transform(df['Severity'])

# Split dataset into features (X) and target (y)
X = df.drop(columns=['Severity'])
y = df['Severity'].values

# Load or build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def predict_severity(user_data):
    user_df = pd.DataFrame([user_data], columns=X.columns)
    user_df['Gender'] = user_df['Gender'].map(gender_mapping)
    user_df['General Symptoms'] = user_df['General Symptoms'].map(symptom_mapping)
    user_df['Symptom Duration'] = user_df['Symptom Duration'].map(symptom_duration_mapping)
    user_df['Onset'] = user_df['Onset'].map(onset_mapping)
    chronic_conditions = user_df['Chronic Conditions'].values[0].split(', ')
    user_df['Chronic Conditions'] = len(chronic_conditions)
    user_df['Allergies'] = 1 if user_df['Allergies'].values[0] == 'Yes' else 0
    user_df['Medications'] = 1 if user_df['Medications'].values[0] == 'Yes' else 0
    user_df['Travel History'] = 1 if user_df['Travel History'].values[0] == 'Yes' else 0
    user_df['Contact with Sick Individuals'] = 1 if user_df['Contact with Sick Individuals'].values[0] == 'Yes' else 0
    user_df['Smoking'] = 1 if user_df['Smoking'].values[0] == 'Yes' else 0
    user_df['Alcohol Consumption'] = alcohol_mapping[user_df['Alcohol Consumption'].values[0]]
    user_df['Physical Activity'] = physical_mapping[user_df['Physical Activity'].values[0]]
    user_df['Stress Levels'] = user_df['Stress Levels'] / 10.0
    user_df['Sleep Quality'] = sleep_mapping[user_df['Sleep Quality'].values[0]]
    user_df[numeric_columns] = scaler.transform(user_df[numeric_columns])
    severity_prediction = model.predict(user_df)
    severity_class = label_encoder.inverse_transform([np.argmax(severity_prediction)])
    return severity_class[0]

st.title("MEDICAL SURVEY PAGE")

if not st.session_state.get('submitted', False):
    st.write("#### Please fill out the following questionnaire to assess severity:")

    # Input fields
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.number_input("Age", min_value=0, value=30)
    pain_scale = st.slider("Rate the  Pain you are facing from 0 to 10", 0, 10, 5)
    general_symptom = st.selectbox("Mention the General Symptom you are facing", list(symptom_mapping.keys()))
    symptom_duration = st.selectbox("What is the Duration of the Symptom", ['Less than 2 days', '2-5 days', 'More than 5 days'])
    onset = st.selectbox("What is the Onset of the symptom you face currently", ['Sudden', 'Gradual'])
    chronic_conditions = st.selectbox("Chronic Conditions you aldready have", ['Diabetes', 'Hypertension', 'Asthma', 'Arthritis', 'Obesity', 'Cholesterol', 'Depression', 'Cirrhosis', 'No chronic conditions'])
    allergies = st.selectbox("Do you have allergies?", ['Yes', 'No'])
    medications = st.selectbox("Are you currently taking medications?", ['Yes', 'No'])
    travel_history = st.selectbox("Did you recently travel anywhere?", ['Yes', 'No'])
    contact_sick = st.selectbox("Did you have Contact with sick individuals?", ['Yes', 'No'])
    smoking = st.selectbox("Do you smoke?", ['Yes', 'No'])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ['No', 'Occasionally', 'Regularly'])
    physical_activity = st.selectbox("Physical Activity", ['No', 'Light', 'Moderate', 'Intense'])
    stress_levels = st.slider("select your Stress Level (0 to 10)", 0, 10, 5)
    sleep_quality = st.selectbox("How is your Sleep Quality?", ['Excellent', 'Good', 'Fair', 'Poor'])

    user_data = {
        'Gender': gender,
        'Age': age,
        'Pain Scale': pain_scale,
        'General Symptoms': general_symptom,
        'Symptom Duration': symptom_duration,
        'Onset': onset,
        'Chronic Conditions': chronic_conditions,
        'Allergies': allergies,
        'Medications': medications,
        'Travel History': travel_history,
        'Contact with Sick Individuals': contact_sick,
        'Smoking': smoking,
        'Alcohol Consumption': alcohol_consumption,
        'Physical Activity': physical_activity,
        'Stress Levels': stress_levels,
        'Sleep Quality': sleep_quality
    }

    # Submit button
    if st.button("Submit"):
        # Save the user data and mark submission
        st.session_state.submitted = True
        st.session_state.user_data = user_data
        st.session_state.severity = predict_severity(user_data)

else:
    # Display predicted severity in a large bold box
    severity = st.session_state.severity
    st.write("### ")
    st.write("### Severity Prediction Result")
    st.markdown(f"""<div style='font-size: 32px; font-weight: bold; padding: 20px; text-align: center; border: 2px solid #209cee; border-radius: 10px;'>Predicted Severity Level<br><div style='font-size: 40px; font-weight: bold; color: #209cee;;'>{severity}</div></div>""",unsafe_allow_html=True)
    st.write("### ")
    def show_prescription(severity):
        if severity == 'Mild':
            st.write("### RECOMENDATIONS:")
            st.write("- Rest without any disturbance.")
            st.write("- Drink lots of water.")
            st.write("- Don't stress too much .It just mild case")
            st.write("- Try to avoid processed food for brief time.")
        elif severity == 'Moderate':
            st.write("### RECOMENDATIONS:")
            st.write("- Rest without any disturbance.")
            st.write("- Drink lots of water.")
            st.write("- Don't stress too much.")
            st.write("- Avoid alcohol and cigarettes if the habit persists")
            st.write("- Don't sit for too long.try to have small walk")
        elif severity == 'Severe':
            st.write("### RECOMENDATIONS:")
            st.write("- Do not take any medication on your own to counter the symptoms")
            st.write("- Continue with your existing medication if any ")
            st.write("- try to get Rest and dont get streesed")
            st.write("- Book an appointment with our doctor by clicking ""Doctor appointment"" for quick services")
        st.markdown("</div>", unsafe_allow_html=True)
    
    show_prescription(severity)
    # Show recommended medications
    
    selected_symptom = st.session_state.user_data['General Symptoms']
    medications_list = symptoms_medications.get(selected_symptom, "No specific medication found")
    #st.write(f"Recommended Medications for {selected_symptom}: **{medications_list}**")

    
        
    # Back button to return to the previous page
    if st.button("Back"):
        st.session_state.submitted = False
        st.session_state.severity = None

    # Create two columns for buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Doctor Appointment"):
            # Redirect to a new page with summary and prescription
            st.session_state.appointment = True
            st.session_state.severity = severity
            st.session_state.meds_list = medications_list
            st.experimental_rerun()

    with col2:
        if st.button("Order Medicine"):
            # Redirect to a new page with invoice-like format
            st.session_state.order = True
            st.session_state.meds_list = medications_list
            st.experimental_rerun()

    # Check if the appointment or order button was pressed
    if st.session_state.get('appointment', False):
        st.session_state.appointment = False
        
        st.write("### Patient Information Summary for appointment confirmation:")
        st.write("###### Check your given details before confirming your appointment with the Doctor")
        for key, value in st.session_state.user_data.items():
            st.markdown(f"<div style='display:flex; justify-content:space-between; padding: 5px 0; border-bottom: 1px solid #ddd;'><span style='font-weight:bold;'>{key}:</span><span>{value}</span></div>", unsafe_allow_html=True)
        st.write("### ")
        st.markdown(
        "<a href='http://surveydoc-production.up.railway.app' target='_blank' "
        "style='display: inline-block; padding: 10px 20px; font-size: 18px; "
        "background-color: #209cee;; color: white; text-decoration: none; "
        "border-radius: 5px; text-align: center;'>BOOK THE APPOINTMENT</a>",
        unsafe_allow_html=True
    )

    if st.session_state.get('order', False):
        st.session_state.order = False
        medications_invoice = pd.DataFrame({
            "Medication": medications_list.split(', '),
            "Nos": [1] * len(medications_list.split(', '))  # Example fixed price per medication
        })
        #medications_invoice['Total'] = medications_invoice['Quantity'] * medications_invoice['Price']
        st.write("### Invoice for Recommended Medications:")
        st.table(medications_invoice)
        st.markdown(
        "<a href='   ' target='_blank' "
        "style='display: inline-block; padding: 10px 20px; font-size: 18px; "
        "background-color: #209cee;; color: white; text-decoration: none; "
        "border-radius: 5px; text-align: center;'>CONFIRM ORDER</a>",
        unsafe_allow_html=True)

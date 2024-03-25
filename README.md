import pandas as pd  
import numpy as np  
from sklearn.preprocessing import LabelEncoder  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense   
   
# Step 1: Read the CSV dataset  df = 
pd.read_csv('/content/Medical diagnosis .csv')   
   
# Step 2: Convert categorical variables to numerical using LabelEncoder  
label_encoder = LabelEncoder()  df['Gender'] = 
label_encoder.fit_transform(df['Gender'])  df['Diabetes'] = 
label_encoder.fit_transform(df['Diabetes'])  df['Hypertension'] = 
label_encoder.fit_transform(df['Hypertension'])  df['Label'] = 
label_encoder.fit_transform(df['Label'])   
   
# Step 3: Extract features and labels from the DataFrame  
data = df[['Age', 'Gender', 'Diabetes', 'Hypertension']].values.astype(np.float32)  labels = 
df['Label'].values.astype(np.float32)  

# Step 4: Define and compile the model 
model = Sequential([   
    Dense(64, activation='relu', input_shape=(data.shape[1],)),   
    Dense(1, activation='sigmoid')   
])   
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])   
   
# Step 5: Train the model 
model.fit(data, labels, epochs=10, batch_size=1)   
   
# Step 6: Make predictions using the trained model   
new_patient_data = np.array([[32, 0, 1, 0]], dtype=np.float32)  # Gender: Female, Diabetes: 
Yes, Hypertension: No  prediction = model.predict(new_patient_data)   
   
if prediction[0][0] > 0.5:   
    print("The patient may not show up for the appointment.")  
else:   
    print("The patient is likely to show up for the appointment.")   

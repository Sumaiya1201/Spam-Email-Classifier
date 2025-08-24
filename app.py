# app.py

import streamlit as st
import pickle

# 1️⃣ Load the saved model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 2️⃣ App title
st.title("Email Spam Classifier")

st.write("""
Enter any message below and click **Predict**.  
The app will tell you if it’s likely **spam** or **ham**, along with a confidence score.
""")

# 3️⃣ Input box for message
input_sms = st.text_area("Enter your message:")

# 4️⃣ Predict button
if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning(" Please enter a message before predicting.")
    else:
        # Transform input using the saved vectorizer
        vector_input = vectorizer.transform([input_sms])
        
        # Predict using the saved model
        result = model.predict(vector_input)[0]
        # Get prediction probability
        prob = model.predict_proba(vector_input)[0]
        spam_prob = prob[list(model.classes_).index('spam')]

        # Display result with confidence
        if result == "spam":
            st.error(f" This looks like SPAM! (Confidence: {spam_prob*100:.2f}%)")
        else:
            st.success(f"This looks like HAM (Not Spam) (Spam Probability: {spam_prob*100:.2f}%)")

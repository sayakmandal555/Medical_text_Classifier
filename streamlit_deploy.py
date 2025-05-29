import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model and preprocessing tools
@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model("Medical_text_Classifier_CSV.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_model_and_tools()

# Define prediction function
def predict_medical_class(text, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_index = tf.argmax(prediction, axis=1).numpy()[0]
    return label_encoder.inverse_transform([predicted_index])[0]

# Streamlit UI
st.title("Medical Text Classifier")
st.write("Enter a medical description to predict its category:")

user_input = st.text_area("Medical Text", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        predicted_class = predict_medical_class(user_input)
        st.success(f"🔍 Predicted Medical Class: **{predicted_class}**")

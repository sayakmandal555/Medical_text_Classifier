from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Input schema using Pydantic
class TextInput(BaseModel):
    text: str

# Initialize FastAPI app
app = FastAPI(title="Medical Text Classifier")

# Load model and preprocessing tools
model = tf.keras.models.load_model("Medical_text_Classifier_CSV.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Prediction logic
def predict_medical_class(text: str, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_index = tf.argmax(prediction, axis=1).numpy()[0]
    return label_encoder.inverse_transform([predicted_index])[0]

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "API is running. Use POST /predict to classify text."}

# Prediction endpoint
@app.get("/predict")
def predict(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    try:
        result = predict_medical_class(input_data.text)
        return {
            "input_text": input_data.text,
            "predicted_class": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

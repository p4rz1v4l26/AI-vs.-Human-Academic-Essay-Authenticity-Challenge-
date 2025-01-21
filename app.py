import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

st.set_page_config(
    page_title="Academic Essay Classifier",
    page_icon="logo.png",  
)

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))  
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

model, tokenizer = load_model()

# Define a prediction function
def predict(sentence):
    inputs = tokenizer(
        sentence,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "AI" if prediction == 1 else "Human"

col1, col2, col3 = st.columns([1, 2, 1])  
with col2:  
    st.image("logo.png", width=100)  

st.markdown(
    """
    <h1 style="text-align: center;">Academic Essay Classifier</h1>
    """,
    unsafe_allow_html=True,
)

st.write("Enter a sentence to check whether it's AI-generated or written by a human.")

sentence = st.text_input("Enter a sentence:")
if st.button("Classify"):
    if sentence.strip():
        result = predict(sentence)
        st.success(f"The given text is classified as: **{result}**")
    else:
        st.error("Please enter a valid sentence.")
# AI-vs.-Human-Academic-Essay-Authenticity-Challenge-
Final year project of Group CST-G09

## **Overview**

The Academic Essay Classifier is a machine learning-based application that determines whether a given text is AI-generated or written by a human. The model is powered by a fine-tuned BERT model and is accessible through a Streamlit web interface.

---

## **Features**

- Classifies text as either **AI-generated** or **Human-written**.
- Utilizes a **fine-tuned BERT model** for text classification.
- Interactive **Streamlit web interface** for easy classification.
- Supports **real-time text input** for classification.

---

## **Technologies Used**

- **Python**  
- **Streamlit**: For the web-based user interface.  
- **Hugging Face Transformers**: For BERT model implementation.  
- **PyTorch**: For model inference.  

---

## **Setup Instructions**

### **Prerequisites**
Ensure the following are installed:
- Python 3.8 or above
- pip (Python package manager)

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/p4rz1v4l26/AI-vs.-Human-Academic-Essay-Authenticity-Challenge-.git
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```
### **Step 3: Download the Pretrained Model**
Make sure you have the `model.pth` file in the project directory. if not pls contact me


### **Step 4: Run the Application**
```bash
streamlit run app.py
```


import streamlit as st
import joblib
import pandas as pd
from src.preprocess import clean_text_v1  # Import hàm tiền xử lý của bạn

# 1. Load model và vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('./results/trained_models/best_svm_model.pkl')
    vectorizer = joblib.load('./results/trained_models/tfidf_vectorizer.pkl')
    return model, vectorizer

# 2. Hàm dự đoán
def predict_spam(message, model, vectorizer):
    # Tiền xử lý
    cleaned_message = clean_text_v1(message)
    # Vector hóa
    message_tfidf = vectorizer.transform([cleaned_message])
    # Dự đoán
    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]
    return prediction, probability

# 3. Giao diện Streamlit
def main():
    st.title("Spam Detection Demo")
    st.write("Nhập tin nhắn để kiểm tra spam/ham")
    
    # Load model
    model, vectorizer = load_model()
    
    # Input
    message = st.text_area("Tin nhắn:", height=150)
    
    if st.button("Kiểm tra"):
        if message:
            # Dự đoán
            prediction, probability = predict_spam(message, model, vectorizer)
            
            # Hiển thị kết quả
            st.write("---")
            st.subheader("Kết quả:")
            
            # Hiển thị nhãn dự đoán với màu sắc
            if prediction == 0:
                st.markdown("<h3 style='color: green;'>Kết quả: HAM</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: red;'>Kết quả: SPAM</h3>", unsafe_allow_html=True)
            
            # Hiển thị xác suất
            st.write(f"Xác suất HAM: {probability[0]:.2%}")
            st.write(f"Xác suất SPAM: {probability[1]:.2%}")
            
            # Hiển thị tin nhắn đã tiền xử lý
            st.write("---")
            st.subheader("Tin nhắn sau tiền xử lý:")
            st.write(clean_text_v1(message))
        else:
            st.warning("Vui lòng nhập tin nhắn!")

if __name__ == "__main__":
    main()

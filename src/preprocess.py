import re
import nltk
# Đảm bảo đã tải stopwords: nltk.download('stopwords')
from nltk.corpus import stopwords

# Khởi tạo danh sách stop words một lần
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("Downloading nltk stopwords...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text_v1(text):
    """
    Pipeline tiền xử lý v1:
    1. Chuyển chữ thường
    2. Xóa dấu câu, số, ký tự đặc biệt (chỉ giữ chữ cái và khoảng trắng)
    3. Tách từ (tokenize đơn giản bằng split)
    4. Xóa stop words
    5. Nối lại thành chuỗi
    """
    # 1. Chuyển chữ thường
    text = text.lower()

    # 2. Xóa ký tự không mong muốn
    text = re.sub(r'[^a-z0-9\s@#$%]', '', text)
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Tách từ
    tokens = text.split()

    # 4. Xóa stop words
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Nối lại
    cleaned_text = " ".join(tokens)

    return cleaned_text

def clean_text_v2(text):
    """
    Pipeline tiền xử lý v2 với cải tiến:
    - Giữ lại số và ký tự đặc biệt quan trọng
    - Sử dụng word_tokenize thay vì split
    - Custom stop words list
    - Thêm lemmatization để chuẩn hóa từ
    """
    # 1. Chuyển chữ thường
    text = text.lower()
    
    # 2. Giữ lại số và ký tự đặc biệt quan trọng
    text = re.sub(r'[^a-z0-9\s@#$%]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Tokenize tốt hơn
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    
    # 4. Custom stop words
    custom_stop_words = stop_words - {'not', 'no', 'never'}
    tokens = [word for word in tokens if word not in custom_stop_words]
    
    # 5. Thêm lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

print(stop_words)
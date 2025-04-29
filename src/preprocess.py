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
    text = re.sub(r'[^a-z\s]', '', text)
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # 3. Tách từ
    tokens = text.split()

    # 4. Xóa stop words
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Nối lại
    cleaned_text = " ".join(tokens)

    return cleaned_text

# Bạn có thể thêm các hàm khác ở đây nếu cần (ví dụ: lemmatization)
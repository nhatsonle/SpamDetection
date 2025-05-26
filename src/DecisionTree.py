# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
# nltk.download('stopwords') # Chạy lần đầu nếu chưa tải
from nltk.corpus import stopwords

# Tắt các cảnh báo không cần thiết (tùy chọn)
import warnings
warnings.filterwarnings('ignore')

# %%
# 1. Đọc dữ liệu
# Dataset này thường cần encoding='latin-1' hoặc 'ISO-8859-1'
try:
    df = pd.read_csv('../data/raw/spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file spam.csv trong thư mục data/raw/")
    # Dừng hoặc xử lý lỗi tại đây
    exit() # Hoặc cách xử lý khác

print("Đọc dữ liệu thành công!")
print("5 dòng đầu tiên:")
print(df.head())

# %%
# 2. Tiền xử lý cột và kiểm tra dữ liệu thiếu
# Thường dataset này có 5 cột, chỉ cần 2 cột đầu
df = df[['Category', 'Message']]

df.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)

print("\nThông tin DataFrame sau khi xử lý cột:")
df.info()

print("\nKiểm tra dữ liệu thiếu:")
print(df.isnull().sum())

# Chuyển đổi nhãn sang dạng số (ví dụ: ham=0, spam=1) - tiện cho mô hình sau này
df['Category_Num'] = df['Category'].map({'ham': 0, 'spam': 1})
print("\n5 dòng đầu sau khi thêm cột Category_Num:")
print(df.head())

# Lưu lại phiên bản này nếu muốn (tùy chọn)

# %%
df.to_csv('C:\\Users\\ADMIN\\Documents\\GitHub\\AI\\SpamDetection\\data\\raw\\processed\\spam_cleaned_columns.csv', index=False)

# %%
# 3. EDA Cơ bản

# Phân bố lớp (Spam vs. Ham)
print("\nPhân bố lớp Spam/Ham:")
print(df['Category'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='Category', data=df)
plt.title('Phân bố số lượng Spam và Ham')
plt.xlabel('Loại tin nhắn')
plt.ylabel('Số lượng')
# Lưu biểu đồ (nếu muốn)
plt.savefig('../results/figures/class_distribution.png')
plt.show()

# %%
# Phân tích độ dài tin nhắn
df['Message_Length'] = df['Message'].apply(len)
print("\nThống kê độ dài tin nhắn:")
print(df['Message_Length'].describe())

plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Message_Length', hue='Category', kde=True, bins=50)
plt.title('Phân bố độ dài tin nhắn theo loại (Spam vs. Ham)')
plt.xlabel('Độ dài tin nhắn')
plt.ylabel('Tần suất')
# plt.savefig('../results/figures/length_distribution.png')
plt.show()

# %%
# Xem xét các tin nhắn có độ dài bất thường (nếu cần)
print("\nTin nhắn dài nhất:")
print(df[df['Message_Length'] == df['Message_Length'].max()]['Message'].iloc[0])
print("\nTin nhắn ngắn nhất:")
print(df[df['Message_Length'] == df['Message_Length'].min()]['Message'].iloc[0])

# %%
# (Tùy chọn) Word Cloud
try:
    from wordcloud import WordCloud

    spam_text = " ".join(df[df['Category'] == 'spam']['Message'])
    ham_text = " ".join(df[df['Category'] == 'ham']['Message'])

    # Word cloud cho Spam
    plt.figure(figsize=(10, 5))
    wordcloud_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
    plt.imshow(wordcloud_spam, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud cho tin nhắn Spam')
    plt.savefig('../results/figures/wordcloud_spam.png')
    plt.show()

    # Word cloud cho Ham
    plt.figure(figsize=(10, 5))
    wordcloud_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
    plt.imshow(wordcloud_ham, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud cho tin nhắn Ham')
    plt.savefig('../results/figures/wordcloud_ham.png')
    plt.show()

except ImportError:
    print("\nThư viện wordcloud chưa được cài đặt. Bỏ qua bước tạo Word Cloud.")
    print("Để cài đặt, chạy: pip install wordcloud")

# %%
# 4. Áp dụng Pipeline tiền xử lý v1

# Import hàm từ file src (đảm bảo file nằm đúng cấu trúc)
import sys
import os
# Thêm thư mục src vào đường dẫn để import
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.preprocess import clean_text_v1

print("\nÁp dụng tiền xử lý v1...")
# Tạo cột mới chứa tin nhắn đã làm sạch
df['Cleaned_Message'] = df['Message'].apply(clean_text_v1)

print("5 dòng đầu với Cleaned_Message:")
print(df[['Message', 'Cleaned_Message']].head())

# Xem thử một vài kết quả
print("\nVí dụ tiền xử lý:")
for i in range(5):
    print(f"Gốc : {df['Message'].iloc[i]}")
    print(f"Sạch: {df['Cleaned_Message'].iloc[i]}\n")

# Lưu lại DataFrame đã xử lý nếu muốn
# df.to_csv('../data/processed/spam_preprocessed_v1.csv', index=False)

# %%
# 5. Bắt đầu triển khai Decision Tree (Baseline - sử dụng entropy)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\dtree Bắt đầu huấn luyện mô hình Decision Tree...")

# Dữ liệu đầu vào cho mô hình là cột Cleaned_Message
X = df['Cleaned_Message']
y = df['Category_Num'] # Nhãn dạng số (0 hoặc 1)

# Chia dữ liệu Train/Test (Tỷ lệ 80/20)
# Dùng random_state=42 để cả nhóm có kết quả chia giống nhau
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Kích thước tập huấn luyện: {X_train.shape[0]}")
print(f"Kích thước tập kiểm tra: {X_test.shape[0]}")

# Khởi tạo TF-IDF Vectorizer
# max_features giới hạn số lượng từ trong từ điển (tùy chọn, giúp giảm chiều dữ liệu)
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Bạn có thể thử nghiệm giá trị khác

# Fit và transform tập huấn luyện
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Chỉ transform tập kiểm tra (dùng từ điển đã học từ tập train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nKích thước ma trận TF-IDF tập huấn luyện: {X_train_tfidf.shape}")
print(f"Kích thước ma trận TF-IDF tập kiểm tra: {X_test_tfidf.shape}")

# Khởi tạo và huấn luyện mô hình Decision Tree
dtree_model = DecisionTreeClassifier(criterion = 'entropy')
dtree_model.fit(X_train_tfidf, y_train)

print("\nĐã huấn luyện xong mô hình Decision Tree.")

# Dự đoán trên tập kiểm tra
y_pred_dtree = dtree_model.predict(X_test_tfidf)

# Đánh giá ban đầu
print("\nKết quả đánh giá Decision Tree (Entropy):")
print("Accuracy:", accuracy_score(y_test, y_pred_dtree))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dtree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dtree, target_names=['ham', 'spam']))

# (Tùy chọn) Lấy xác suất
# y_pred_proba_dtree = dtree_model.predict_proba(X_test_tfidf)
# print("\nXác suất dự đoán cho 5 mẫu đầu tiên:\n", y_pred_proba_dtree[:5])

# Vẽ cây quyết định với max_depth = 3 để minh hoạ cho việc phân nhanh của cây
from sklearn import tree
fig = plt.figure(figsize=(10,5))

_ = tree.plot_tree(dtree_model,filled = True, max_depth = 3, feature_names = tfidf_vectorizer.get_feature_names_out())

# %%
# 6. Triển khai Decision Tree (sử dụng gini)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\dtree Bắt đầu huấn luyện mô hình Decision Tree...")

# Dữ liệu đầu vào cho mô hình là cột Cleaned_Message
X = df['Cleaned_Message']
y = df['Category_Num'] # Nhãn dạng số (0 hoặc 1)

# Chia dữ liệu Train/Test (Tỷ lệ 80/20)
# Dùng random_state=42 để cả nhóm có kết quả chia giống nhau
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Kích thước tập huấn luyện: {X_train.shape[0]}")
print(f"Kích thước tập kiểm tra: {X_test.shape[0]}")

# Khởi tạo TF-IDF Vectorizer
# max_features giới hạn số lượng từ trong từ điển (tùy chọn, giúp giảm chiều dữ liệu)
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Bạn có thể thử nghiệm giá trị khác

# Fit và transform tập huấn luyện
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Chỉ transform tập kiểm tra (dùng từ điển đã học từ tập train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nKích thước ma trận TF-IDF tập huấn luyện: {X_train_tfidf.shape}")
print(f"Kích thước ma trận TF-IDF tập kiểm tra: {X_test_tfidf.shape}")

# Khởi tạo và huấn luyện mô hình Decision Tree
dtree_model = DecisionTreeClassifier(criterion = 'gini')
dtree_model.fit(X_train_tfidf, y_train)

print("\nĐã huấn luyện xong mô hình Decision Tree.")

# Dự đoán trên tập kiểm tra
y_pred_dtree = dtree_model.predict(X_test_tfidf)

# Đánh giá ban đầu
print("\nKết quả đánh giá Decision Tree (Gini Index):")
print("Accuracy:", accuracy_score(y_test, y_pred_dtree))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_dtree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dtree, target_names=['ham', 'spam']))

# (Tùy chọn) Lấy xác suất
# y_pred_proba_dtree = dtree_model.predict_proba(X_test_tfidf)
# print("\nXác suất dự đoán cho 5 mẫu đầu tiên:\n", y_pred_proba_dtree[:5])

# Vẽ cây quyết định với max_depth = 3 để minh hoạ cho việc phân nhanh của cây
from sklearn import tree
fig = plt.figure(figsize=(10,5))

_ = tree.plot_tree(dtree_model,filled = True, max_depth = 3, feature_names = tfidf_vectorizer.get_feature_names_out())



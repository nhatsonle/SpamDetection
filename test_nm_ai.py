import pandas as pd 

dataFrame = pd.read_csv(r"C:\Users\HP\Downloads\SpamDetection\data\raw\spam.csv") 

print ("Our Dataframe ....",dataFrame)

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
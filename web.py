# web.py
import streamlit as st
from model import RNN
import torch
import pickle

# 加載 token_to_num 詞典
with open('token_to_num.pkl', 'rb') as f:
    token_to_num = pickle.load(f)

# 加載模型
model = RNN(embedding_dim=256, hidden_size=64, num_layer=2, token_to_num=token_to_num)
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
model.eval()

st.title('情感分析模型')

# 文本輸入
review = st.text_area("輸入您的評論：")

if st.button('預測'):
    # 進行預測
    prediction = model.predict(review)
    if prediction == 1:
        st.write('正面')
    else:
        st.write('負面')

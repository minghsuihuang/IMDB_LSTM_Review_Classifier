import streamlit as st
from model import Model
import torch
import pickle

# 載入先前儲存的字典，這個字典是用來將文字轉換成模型理解的數字格式
with open('token_to_num.pkl', 'rb') as f:
    token_to_num = pickle.load(f)

# 載入模型
model = Model(embedding_dim=256, hidden_size=64, num_layer=2, token_to_num=token_to_num)
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
model.eval()

# 設定頁面的外觀
st.set_page_config(layout="wide", page_title="Sentiment Analysis", page_icon=":smiley:")

# 利用 Streamlit 的 markdown 功能來自定義 CSS 樣式
st.markdown(
    """
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """,
    unsafe_allow_html=True)

st.title('情感分析模型') # 標題

col1, col2 = st.columns(2)

with col1:
    st.markdown('## 請輸入要預測的字句') 
    # 可以在這裡輸入想要分析情緒的文字
    review = st.text_area("請輸入您要分析情緒的文字：", height=150)

with col2:
    st.markdown('## 預測') # 預測結果顯示的地方
    if st.button('預測'):
        # 顯示一個小圈圈讓用戶知道模型正在分析情緒
        with st.spinner('正在分析情緒...'):
            prediction = model.predict(review) 
            # 根據模型的預測結果來顯示正面還是負面情緒
            if prediction == 1:
                st.success('正面情緒 🙂') # 用綠色顯示正面
            else:
                st.error('負面情緒 🙁') # 用紅色顯示負面

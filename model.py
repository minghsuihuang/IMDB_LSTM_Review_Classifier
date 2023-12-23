# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layer, token_to_num):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.token_to_num = token_to_num  # 新增

        self.embedding = nn.Embedding(127561, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layer, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 4, 20)
        self.fc1 = nn.Linear(20, 2)

    def preprocess_text(self, sentence):
        sentence = re.sub(r'<[^>]+>', ' ', sentence)
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
    
        return sentence.lower()

    def forward(self, x):
        x = self.embedding(x)
        states, hidden = self.lstm(x.permute([1,0,2]), None)
        x = torch.cat((states[0], states[-1]), 1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(x)
        return x

    def predict(self, review, max_len=500):
        # 處理文本
        processed_review = self.preprocess_text(review)

        # 轉換為數字表示
        num_review = [self.token_to_num.get(word, 0) for word in processed_review.split()]

        # 添加填充或截斷
        if len(num_review) < max_len:
            num_review += [0] * (max_len - len(num_review))
        else:
            num_review = num_review[:max_len]

        # 做出預測
        with torch.no_grad():
            data = torch.tensor([num_review]).to(device)
            output = self(data)
            prediction = torch.argmax(output, dim=1)
        return prediction.item()
    
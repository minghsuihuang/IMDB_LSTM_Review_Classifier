# main.py
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
import numpy as np
from tqdm import tqdm
import torch.utils.data as data
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from model import RNN


# 檢查是否有可用的GPU，否則使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDB(Dataset):
    def __init__(self, data, max_len=500):
        self.data = []
        reviews = data['review'].tolist()
        sentiments = data['sentiment'].tolist()
        reviews, max_len, self.token_to_num = self.get_token2num_maxlen(reviews)
        max_len = 500
        
        for review, sentiment in zip(reviews, sentiments):
            if max_len > len(review):
                padding_cnt = max_len - len(review)
                review += padding_cnt * [0]
            else:
                review = review[:max_len]

            if sentiment == 'positive':
                label = 1
            else:
                label = 0

            self.data.append([review, label])

    def __getitem__(self, index):
        datas = torch.tensor(self.data[index][0])
        labels = torch.tensor(self.data[index][1])
        
        return datas, labels
    
    def __len__(self):
        return len(self.data)
        
    def preprocess_text(self, sentence):
        sentence = re.sub(r'<[^>]+>', ' ', sentence)
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
    
        return sentence.lower()

    
    def get_token2num_maxlen(self, reviews, enable=True):
        token = []
        for review in reviews:
            review = self.preprocess_text(review)
            token += review.split(' ')
        
        token_to_num = {data: cnt for cnt, data in enumerate(list(set(token)), 1)}
         
        num = []
        max_len = 0 
        for review in reviews:
            review = self.preprocess_text(review)
            tmp = []
            for token in review.split(' '):
                tmp.append(token_to_num[token])
                
            if len(tmp) > max_len:
                max_len = len(tmp)
            num.append(tmp)
            
        return num, max_len, token_to_num  # 返回 token_to_num


def train(train_loader, test_loader, model, optimizer, criterion):
    epochs = 10
    train_losses, train_accs, test_accs = [], [], []
    
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')

        model.train()
        for cnt, (data, label) in enumerate(train_iter, 1):
            data, label = data.to(device), label.to(device)
            outputs = model(data)
            loss = criterion(outputs, label)
            _, predict_label = torch.max(outputs, 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (predict_label == label).sum().item()

            # 更新進度條的描述和後綴
            train_iter.set_postfix(loss=train_loss / cnt, acc=train_acc / (cnt * data.size(0)))

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        test_loss = 0
        test_acc = 0
        test_iter = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [Test]')
        model.eval()
        with torch.no_grad():
            for cnt, (data, label) in enumerate(test_iter, 1):
                data, label = data.to(device), label.to(device)
                outputs = model(data)
                loss = criterion(outputs, label)
                _, predict_label = torch.max(outputs, 1)
                test_loss += loss.item()
                test_acc += (predict_label == label).sum().item()

                # 更新進度條的描述和後綴
                test_iter.set_postfix(loss=test_loss / cnt, acc=test_acc / (cnt * data.size(0)))

        avg_test_acc = test_acc / len(test_loader.dataset)
        test_accs.append(avg_test_acc)

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Test Acc: {avg_test_acc:.4f}')

    return train_losses, train_accs, test_accs


# 讀取數據
df = pd.read_csv('IMDB Dataset.csv')

# 數據處理
dataset = IMDB(df)
_, _, token_to_num = dataset.get_token2num_maxlen(df['review'].tolist())
with open('token_to_num.pkl', 'wb') as f:
    pickle.dump(token_to_num, f)
train_set_size = int(len(dataset) * 0.8)
test_set_size = len(dataset) - train_set_size
train_set, test_set = data.random_split(dataset, [train_set_size, test_set_size])

# DataLoader
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)

# 加載 token_to_num 字典
with open('token_to_num.pkl', 'rb') as f:
    token_to_num = pickle.load(f)

# 創建模型實例
model = RNN(embedding_dim=256, hidden_size=64, num_layer=2, token_to_num=token_to_num).to(device)
optimizer = opt.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 訓練模型並獲得結果
train_losses, train_accs, test_accs = train(train_loader, test_loader, model, optimizer, criterion)

# 繪製訓練過程中的損失和準確率圖
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 保存模型
# torch.save(model.state_dict(), 'model.pt')

model_save_path = 'model.pt'
torch.save(model.state_dict(), model_save_path)

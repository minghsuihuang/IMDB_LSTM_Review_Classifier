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
from torch.optim.lr_scheduler import StepLR
from model import Model

# 檢查是否有可用的GPU，如果沒有，就用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義一個數據集類，用於IMDB影評數據
class IMDB(Dataset):
    def __init__(self, data, max_len=500):
        self.data = []
        reviews = data['review'].tolist()
        sentiments = data['sentiment'].tolist()

        # 處理文本數據，將其轉換為數字表示
        reviews, max_len, self.token_to_num = self.get_token2num_maxlen(reviews)
        max_len = 500
        
        # 對每條評論進行填充或截斷，使其長度一致
        for review, sentiment in zip(reviews, sentiments):
            if max_len > len(review):
                padding_cnt = max_len - len(review)
                review += padding_cnt * [0]
            else:
                review = review[:max_len]

            # 將情感標籤轉換為數字
            if sentiment == 'positive':
                label = 1
            else:
                label = 0

            self.data.append([review, label])

    def __getitem__(self, index):
        # 獲取單個數據樣本
        datas = torch.tensor(self.data[index][0])
        labels = torch.tensor(self.data[index][1])
        return datas, labels
    
    def __len__(self):
        # 數據集的長度
        return len(self.data)
        
    # 文本預處理函數
    def preprocess_text(self, sentence):
        sentence = re.sub(r'<[^>]+>', ' ', sentence) # 移除HTML標籤
        sentence = re.sub('[^a-zA-Z]', ' ', sentence) # 保留字母
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence) # 移除單個字母
        sentence = re.sub(r'\s+', ' ', sentence) # 移除多餘空格
        return sentence.lower() # 轉換為小寫
    
    # 將文本轉換為數字表示
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
            
        return num, max_len, token_to_num

# 定義訓練函數
def train(train_loader, test_loader, model, optimizer, criterion, model_save_path): #, scheduler):
    epochs = 200
    train_losses, train_accs, test_accs = [], [], []
    best_acc = 0.0 
    
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

            train_iter.set_postfix(loss=train_loss / cnt, acc=train_acc / (cnt * data.size(0)))

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # 更新學習率
        #scheduler.step()

        # 開始測試階段
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

                test_iter.set_postfix(loss=test_loss / cnt, acc=test_acc / (cnt * data.size(0)))

        avg_test_acc = test_acc / len(test_loader.dataset)
        test_accs.append(avg_test_acc)

        # 打印訓練結果
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Test Acc: {avg_test_acc:.4f}')

        # 保存最好的模型
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_acc:.4f}')

    return train_losses, train_accs, test_accs

# 讀取IMDB數據集
df = pd.read_csv('IMDB_Dataset.csv')

# 數據處理
dataset = IMDB(df)
_, _, token_to_num = dataset.get_token2num_maxlen(df['review'].tolist())
# 將token到數字的映射保存到文件，方便之後使用
with open('token_to_num.pkl', 'wb') as f:
    pickle.dump(token_to_num, f)

# 切分數據集為訓練集和測試集
train_set_size = int(len(dataset) * 0.8)  # 80%作為訓練集
test_set_size = len(dataset) - train_set_size  # 剩下的作為測試集
train_set, test_set = data.random_split(dataset, [train_set_size, test_set_size])

# 創建DataLoader來加載數據集，方便批量處理
train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True, num_workers=0)

# 從文件加載token到數字的映射
with open('token_to_num.pkl', 'rb') as f:
    token_to_num = pickle.load(f)

# 初始化模型、優化器、學習率調度器和損失函數
model = Model(embedding_dim=256, hidden_size=64, num_layer=2, token_to_num=token_to_num).to(device)
optimizer = opt.Adam(model.parameters(), lr=0.01)
#scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # 每50個epoch調整學習率
criterion = nn.CrossEntropyLoss()

# 訓練模型並保存最佳模型
model_save_path = 'best_model.pt'
train_losses, train_accs, test_accs = train(train_loader, test_loader, model, optimizer, criterion, model_save_path)#, scheduler)

# 繪製訓練和測試過程中的損失和準確率
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

# 保存最終模型
model_save_path = 'final_model.pt'
torch.save(model.state_dict(), model_save_path)

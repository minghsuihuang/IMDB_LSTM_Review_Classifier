# IMDB_LSTM_Review_Classifier

IMDB_LSTM_Review_Classifier 是一個基於長短期記憶網絡（LSTM）的情感分析模型，專門用於分析和預測 IMDB 電影評論的情感極性（正面或負面）。

## 特點

- 使用 LSTM 模型來分析文本數據。
- 能夠處理和分類大量的 IMDB 電影評論數據。
- 包括數據預處理、模型訓練、評估和預測功能。
- 附帶一個基於 Streamlit 的 Web 應用，用於實時情感分析。

## 環境要求

- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Streamlit
- tqdm

## 安裝指南

首先，克隆此存儲庫：

git clone https://github.com/yourusername/IMDB_LSTM_Review_Classifier.git


## 使用說明

1. 使用提供的數據集對模型進行訓練或加載已訓練的模型。
2. 運行 Streamlit 應用進行實時評論情感分析。

3. 在應用中輸入評論，獲取情感分析結果。

## 模型結構

詳細的模型結構和參數設置可在 `model.py` 文件中找到。

## 貢獻指南

如果您想對此項目作出貢獻，請遵循以下步驟：

1. Fork 此存儲庫。
2. 創建一個新的分支：`git checkout -b new-feature`。
3. 提交您的更改：`git commit -am 'Add some feature'`。
4. 推送到分支：`git push origin new-feature`。
5. 提交一個 Pull Request。

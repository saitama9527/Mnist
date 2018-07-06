MNIST 手寫數字辨識資料集是機器學習領域最有名的資料集之一, 包含了 70000 個 0~9 手寫阿拉伯數字的 BMP 格式圖檔與其正確之標籤 (Label, 即圖檔對應之 0~9 數字), 其中前 60000 個為訓練樣本, 用來訓練神經網路以建立模型; 其餘 10000 個為測試樣本, 用來檢驗模型是否能正確推論或預測圖片中的數字. 資料集的每個圖片都是解析度為 28*28 (784 個 pixel) 的灰階影像, 每個像素為 0~255 之數值.

首先導入keras內建的mnist資料庫作為data
![Alt text](https://imgur.com/XpqdvLc.jpg)
建立CNN模型
![Alt text](https://imgur.com/B4hiwpe.jpg)
最後compile/訓練 使用Adam作為lr等數據的優化器
![Alt text](https://imgur.com/2WusJxp.jpg)
結果
![Alt text](https://imgur.com/E2DVK6C.jpg)

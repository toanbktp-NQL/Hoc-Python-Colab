# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:20:10 2026

@author: Dell User
"""

# 1. Khai báo thư viện
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 2. Tải dữ liệu hoa Iris có sẵn
iris = load_iris()
X = iris.data  # Kích thước các bộ phận của hoa
y = iris.target # Tên loại hoa (0, 1, hoặc 2)

# 3. Chia dữ liệu thành 2 phần: Học và Kiểm tra
# Chúng ta dùng 80% dữ liệu để dạy máy, và 20% để xem máy đoán đúng không
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Khởi tạo và huấn luyện mô hình (Dùng Logistic Regression để phân loại)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5. Máy thực hiện dự đoán trên phần dữ liệu Kiểm tra
y_pred = model.predict(X_test)

# 6. Kiểm tra xem máy đoán đúng bao nhiêu %
accuracy = accuracy_score(y_test, y_pred)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# 7. Thử đưa thông số 1 bông hoa mới cho máy đoán
# Giả sử hoa có: Dài đài hoa=5.1, Rộng đài hoa=3.5, Dài cánh hoa=1.4, Rộng cánh hoa=0.2
hoa_moi = [[5.1, 3.5, 1.4, 0.2]]
ket_qua = model.predict(hoa_moi)
ten_hoa = iris.target_names[ket_qua[0]]
print(f"Dự đoán loại hoa cho thông số trên là: {ten_hoa}")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# 1. Tải dữ liệu
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['species'] = [iris_data.target_names[i] for i in iris_data.target]

# 2. Vẽ biểu đồ cặp (Đã sửa lỗi palette)
print("Đang tạo biểu đồ cặp... (Vui lòng đợi)")
# Tôi đổi sang bảng màu 'husl' rất phổ biến và đẹp
sns.pairplot(df, hue='species', palette='husl')
plt.suptitle("Biểu đồ so sánh từng cặp thông số của 3 loài hoa Iris", y=1.02)
plt.show()

# 3. Vẽ biểu đồ 2D
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', s=100, palette='husl')
plt.title("Sự phân cụm rõ rệt dựa trên Cánh hoa (Petal)")
plt.grid(True)
plt.show()
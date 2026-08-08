# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:08:08 2026

@author: Dell User
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Tạo dữ liệu mẫu (Dataset)
# x: Số giờ học (từ 1 giờ đến 100 giờ)
# y: Điểm số (tăng nhanh lúc đầu và chậm dần)
x = np.array([1, 2, 5, 10, 20, 40, 60, 80, 100]).reshape(-1, 1)
y = np.array([10, 25, 45, 60, 72, 82, 88, 92, 95])

# 2. Biến đổi dữ liệu x sang dạng Logarithm
# Đây là bước quan trọng nhất để máy tính hiểu được quy luật đường cong
x_log = np.log(x)

# 3. Khởi tạo và huấn luyện mô hình
# Bản chất vẫn dùng thuật toán Linear nhưng áp dụng trên x đã log-hóa
model = LinearRegression()
model.fit(x_log, y)

# 4. Dự đoán thử
# Nếu bạn học 150 giờ, điểm số sẽ là bao nhiêu?
gio_hoc_moi = np.array([[150]])
du_doan = model.predict(np.log(gio_hoc_moi))
print(f"Dự đoán điểm số nếu học 150 giờ: {du_doan[0]:.2f} điểm")

# 5. Vẽ đồ thị
plt.scatter(x, y, color='blue', label='Dữ liệu thực tế')

# Tạo một đường cong mượt mà để vẽ đường dự đoán
x_curve = np.linspace(1, 150, 100).reshape(-1, 1)
y_curve = model.predict(np.log(x_curve))

plt.plot(x_curve, y_curve, color='green', label='Đường hồi quy Logarithm')
plt.xlabel('Số giờ học')
plt.ylabel('Điểm số')
plt.title('Mối quan hệ giữa Giờ học và Điểm số')
plt.legend()
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:30:49 2026

@author: Dell User
"""

# 1. Khai báo các thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 2. Tạo dữ liệu mẫu (Dataset đơn giản)
# Giả sử: x là diện tích (m2), y là giá nhà (tỷ đồng)
# Dữ liệu này chúng ta tự tạo ra để thực hành
x = np.array([30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
y = np.array([1.2, 1.6, 2.1, 2.4, 2.9, 3.2, 3.8, 4.0])
# 3. Khởi tạo mô hình Hồi quy tuyến tính
model = LinearRegression()
# 4. Huấn luyện mô hình (Học từ dữ liệu)
model.fit(x, y)
# 5. Thử dự đoán giá của một căn nhà có diện tích 120m2
dien_tich_moi = [[120]]
gia_du_doan = model.predict(dien_tich_moi)
print(f"Giá nhà dự đoán cho 120m2 là: {gia_du_doan[0]:.2f} tỷ đồng")
# 6. Vẽ đồ thị để quan sát
plt.scatter(x, y, color='blue', label='Dữ liệu thực tế') # Vẽ các điểm dữ liệu
plt.plot(x, model.predict(x), color='red', label='Đường dự đoán') # Vẽ đường thẳng hồi quy
plt.xlabel('Diện tích (m2)')
plt.ylabel('Giá nhà (tỷ đồng)')
plt.legend()
plt.show()
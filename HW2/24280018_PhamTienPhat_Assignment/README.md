# 24280018 – Phạm Tiến Phát | HW2: Medical Insurance Cost Prediction

## Mô tả bài toán

Dự đoán chi phí bảo hiểm y tế (`charges`) dựa trên thông tin cá nhân của khách hàng (tuổi, giới tính, BMI, số con, tình trạng hút thuốc, vùng địa lý).

**Loại bài toán:** Regression  
**Dataset:** Medical Insurance Cost Dataset (1338 mẫu, 7 cột)

## Cách chạy

```bash
# Cài thư viện
pip install -r requirements.txt

# Chạy notebook
jupyter notebook notebook.ipynb
```

## Cấu trúc thư mục

```
24280018_PhamTienPhat_Assignment/
├── README.md
├── report.md
├── notebook.ipynb
├── requirements.txt
├── data/
│   └── Medical_Insurance_Cost.csv
└── outputs/
    ├── figures/       # Biểu đồ EDA và đánh giá mô hình
    └── results.csv    # Bảng so sánh kết quả các mô hình
```

## Mô hình sử dụng

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. K-Nearest Neighbors Regressor
5. Support Vector Regressor (SVR)

## Kết quả tốt nhất

Random Forest Regressor cho kết quả tốt nhất:
- **R² = 0.8849** trên tập test
- **MAE = 2.591,62 USD**
- **RMSE = 4.599,92 USD**

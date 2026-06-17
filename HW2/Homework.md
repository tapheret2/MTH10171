# Bài tập thực hành cá nhân

## Xây dựng mô hình Machine Learning cho bài toán Classification hoặc Regression

**Môn học:** Nhập môn Khoa học Dữ liệu  
**Hình thức:** Cá nhân  
**Thời hạn nộp:** 15/06/2026

---

## 1. Mục tiêu

Sinh viên vận dụng các kiến thức đã học để thực hiện một quy trình Machine Learning hoàn chỉnh, bao gồm:

- Khám phá dữ liệu (Exploratory Data Analysis - EDA).
- Tiền xử lý dữ liệu (Data Preprocessing).
- Huấn luyện mô hình (Model Training).
- Đánh giá mô hình (Model Evaluation).
- Phân tích và so sánh kết quả.
- Viết báo cáo kết luận dựa trên số liệu thực nghiệm.

---

## 2. Yêu cầu chung

Sinh viên chọn **01 trong 02 dạng bài toán**:

- Classification.
- Regression.

Mỗi sinh viên chọn **01 dataset** trong danh sách được cung cấp bên dưới.

Sinh viên cần tự thực hiện đầy đủ quy trình:

1. Tìm hiểu bài toán và biến mục tiêu.
2. Khám phá dữ liệu.
3. Tiền xử lý dữ liệu.
4. Huấn luyện ít nhất 03 mô hình.
5. Đánh giá mô hình bằng các chỉ số phù hợp.
6. So sánh kết quả.
7. Viết nhận xét và đề xuất cải thiện.

---

## 3. Lựa chọn bài toán

### Lựa chọn 1. Classification

Sinh viên chọn **01 dataset** trong danh sách sau:

| Dataset | Mục tiêu |
|---|---|
| Heart Disease Dataset | Dự đoán bệnh nhân có mắc bệnh tim hay không |
| Telco Customer Churn Dataset | Dự đoán khách hàng có rời bỏ dịch vụ hay không |
| IBM Employee Attrition Dataset | Dự đoán nhân viên có nghỉ việc hay không |
| German Credit Dataset | Dự đoán khách hàng có thuộc nhóm tín dụng tốt hay không |

Với bài toán Classification, biến mục tiêu là một nhãn rời rạc, ví dụ:

- Có/Không.
- Yes/No.
- Churn/Not Churn.
- Good Credit/Bad Credit.

### Lựa chọn 2. Regression

Sinh viên chọn **01 dataset** trong danh sách sau:

| Dataset | Mục tiêu |
|---|---|
| California Housing Dataset | Dự đoán giá nhà trung bình |
| Medical Insurance Cost Dataset | Dự đoán chi phí bảo hiểm y tế |
| Student Performance Dataset | Dự đoán điểm cuối kỳ của sinh viên |
| Automobile Price Dataset | Dự đoán giá xe ô tô |

Với bài toán Regression, biến mục tiêu là một giá trị số liên tục, ví dụ:

- Giá nhà.
- Chi phí bảo hiểm.
- Điểm số.
- Giá xe.

---

## 4. Nội dung thực hiện

### Bước 1. Khám phá dữ liệu (EDA)

### Bước 2. Tiền xử lý dữ liệu

Sinh viên cần xử lý phù hợp với dataset đã chọn:

- Xử lý giá trị thiếu.
- Xử lý dòng trùng lặp.
- Xử lý outlier nếu cần.
- Mã hóa biến phân loại.
- Chuẩn hóa hoặc scale dữ liệu số nếu mô hình yêu cầu.
- Tách dữ liệu thành `X` và `y`.
- Chia dữ liệu thành train/test.

Yêu cầu:

- Với Classification, nên dùng `stratify=y` khi chia train/test nếu dữ liệu mất cân bằng lớp.
- Với Regression, cần tránh để dữ liệu test tham gia vào quá trình fit scaler/encoder.

### Bước 3. Xây dựng mô hình

#### Đối với Classification

Huấn luyện **ít nhất 03 mô hình** trong các mô hình sau:

- Logistic Regression.
- K-Nearest Neighbors (KNN).
- Decision Tree Classifier.
- Random Forest Classifier.
- Support Vector Machine (SVM).
- Naive Bayes.

#### Đối với Regression

Huấn luyện **ít nhất 03 mô hình** trong các mô hình sau:

- Linear Regression.
- Decision Tree Regressor.
- Random Forest Regressor.
- K-Nearest Neighbors Regressor.
- Support Vector Regressor (SVR).

Yêu cầu:

- Mỗi mô hình cần được huấn luyện trên cùng tập train.
- Mỗi mô hình cần được đánh giá trên cùng tập test.
- Nên đặt `random_state` nếu mô hình có tham số này để kết quả có thể lặp lại.

### Bước 4. Đánh giá mô hình

#### Classification

Báo cáo các chỉ số:

- Accuracy.
- Precision.
- Recall.
- F1-score.

Bắt buộc vẽ:

- Confusion Matrix.

Khuyến khích thêm:

- Classification report.
- ROC Curve và ROC-AUC nếu phù hợp.
- Learning curve hoặc validation curve nếu có thời gian.

#### Regression

Báo cáo các chỉ số:

- MAE (Mean Absolute Error).
- MSE (Mean Squared Error).
- RMSE (Root Mean Squared Error).
- R2 Score.

Khuyến khích thêm:

- Biểu đồ Actual vs Predicted.
- Biểu đồ residual.
- Feature importance nếu dùng mô hình cây.
- Learning curve hoặc validation curve nếu có thời gian.

### Bước 5. So sánh và phân tích kết quả

Sinh viên cần lập bảng so sánh kết quả giữa các mô hình.

Phần phân tích cần trả lời:

- Mô hình nào cho kết quả tốt nhất?
- Dựa trên chỉ số nào để chọn mô hình tốt nhất?
- Vì sao mô hình đó có thể hoạt động tốt hơn các mô hình còn lại?
- Có dấu hiệu overfitting hoặc underfitting không?
- Những khó khăn gặp phải trong quá trình thực hiện.
- Hướng cải thiện mô hình trong tương lai.

---

## 5. Yêu cầu nộp bài
Sinh viên nộp theo gợi ý bên dưới hoặc nộp file .ipynb 

Sinh viên nộp một thư mục hoặc file nén có cấu trúc gợi ý:

```text
StudentID_FullName_Assignment/
├── README.md
├── report.pdf hoặc report.md
├── notebook.ipynb hoặc main.py
├── requirements.txt
├── data/
│   └── dataset.csv
└── outputs/
    ├── figures/
    └── results.csv
```

Trong đó:

- `README.md`: hướng dẫn cách chạy bài.
- `report.pdf` hoặc `report.md`: báo cáo kết quả.
- `notebook.ipynb` hoặc `main.py`: mã nguồn thực hiện.
- `requirements.txt`: danh sách thư viện sử dụng.
- `data/`: chứa dataset hoặc hướng dẫn tải dataset.
- `outputs/`: chứa bảng kết quả và hình ảnh minh họa.

Lưu ý:

- Không nộp chỉ mỗi file code mà không có báo cáo.
- Không nộp báo cáo chỉ có hình mà không có nhận xét.
- Không sao chép nguyên văn bài của bạn khác.
- Nếu sử dụng nguồn tham khảo, cần ghi rõ nguồn.

---

## 6. Nội dung báo cáo

Báo cáo nên gồm các phần:

1. Giới thiệu bài toán.
2. Mô tả dataset.
3. Khám phá dữ liệu.
4. Tiền xử lý dữ liệu.
5. Mô hình sử dụng.
6. Kết quả đánh giá.
7. So sánh mô hình.
8. Kết luận và hướng cải thiện.


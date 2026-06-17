# Nhận xét kết quả cuối bài

## 1. Tổng quan kết quả

Sau khi chạy thực nghiệm trên tập Loan Approval Prediction Dataset, các mô hình được đánh giá trên tập test bằng Accuracy, Precision, Recall, F1-score và ROC-AUC.

Bảng kết quả chính:

| Mô hình | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9824 | 0.9831 | 0.9887 | 0.9859 | 0.9987 |
| Decision Tree No Limit | 0.9789 | 0.9831 | 0.9831 | 0.9831 | 0.9776 |
| Decision Tree | 0.9684 | 0.9828 | 0.9661 | 0.9744 | 0.9884 |
| Logistic Regression | 0.9145 | 0.9210 | 0.9435 | 0.9321 | 0.9726 |

Nhìn chung, tất cả mô hình đều có kết quả khá tốt. `Logistic Regression` là mô hình nền tảng ổn định, còn các mô hình cây cho kết quả cao hơn rõ rệt. Trong đó, `Random Forest` là mô hình tốt nhất theo `Test F1` và `Test ROC-AUC`.

---

## 2. Mô hình tốt nhất

Mô hình tốt nhất là **Random Forest**.

Các chỉ số chính:

- `Test Accuracy = 0.9824`
- `Test Precision = 0.9831`
- `Test Recall = 0.9887`
- `Test F1 = 0.9859`
- `Test ROC-AUC = 0.9987`

Diễn giải:

- Accuracy 98.24% cho thấy mô hình dự đoán đúng phần lớn hồ sơ trong tập test.
- Precision 98.31% nghĩa là trong các hồ sơ được mô hình dự đoán `Approved`, đa số thật sự thuộc lớp `Approved`.
- Recall 98.87% nghĩa là mô hình tìm được gần như toàn bộ hồ sơ thật sự `Approved`.
- F1-score 98.59% cho thấy mô hình cân bằng rất tốt giữa Precision và Recall.
- ROC-AUC 99.87% cho thấy mô hình phân biệt hai lớp `Approved` và `Rejected` rất tốt.

Với bài toán phê duyệt khoản vay, Random Forest là lựa chọn phù hợp nhất trong các mô hình đã thử vì vừa có khả năng dự đoán tốt, vừa giảm overfitting tốt hơn so với một Decision Tree đơn lẻ.

---

## 3. So sánh từng mô hình

### Logistic Regression

Kết quả:

- `Test Accuracy = 0.9145`
- `Test Precision = 0.9210`
- `Test Recall = 0.9435`
- `Test F1 = 0.9321`
- `Test ROC-AUC = 0.9726`

Nhận xét:

- Logistic Regression có kết quả thấp nhất trong bốn mô hình, nhưng vẫn là một baseline tốt.
- Train F1 là 0.9345 và Test F1 là 0.9321, hai giá trị rất gần nhau. Điều này cho thấy mô hình ít bị overfitting.
- Mô hình có Recall cao hơn Precision, nghĩa là mô hình có xu hướng tìm được nhiều hồ sơ `Approved`, nhưng vẫn còn một số trường hợp phê duyệt nhầm.
- Vì Logistic Regression là mô hình tuyến tính, nó có thể chưa mô tả tốt các quan hệ phi tuyến trong dữ liệu khoản vay.

### Decision Tree với `max_depth=5`

Kết quả:

- `Test Accuracy = 0.9684`
- `Test Precision = 0.9828`
- `Test Recall = 0.9661`
- `Test F1 = 0.9744`
- `Test ROC-AUC = 0.9884`

Nhận xét:

- Decision Tree cải thiện rõ rệt so với Logistic Regression.
- Precision cao 0.9828 cho thấy mô hình khá thận trọng khi dự đoán hồ sơ `Approved`.
- Recall 0.9661 thấp hơn Precision, nghĩa là vẫn có một số hồ sơ thật sự `Approved` bị dự đoán nhầm thành `Rejected`.
- Train F1 là 0.9790 và Test F1 là 0.9744, khoảng cách nhỏ, nên cây với `max_depth=5` chưa bị overfitting nghiêm trọng.

### Decision Tree No Limit

Kết quả:

- `Train Accuracy = 1.0000`
- `Test Accuracy = 0.9789`
- `Train F1 = 1.0000`
- `Test F1 = 0.9831`
- `Test ROC-AUC = 0.9776`

Nhận xét:

- Mô hình đạt điểm tuyệt đối trên tập train, cho thấy cây đã học rất sát dữ liệu huấn luyện.
- Test F1 vẫn cao, nhưng Test ROC-AUC thấp hơn Random Forest và Decision Tree có giới hạn.
- Đây là dấu hiệu cần lưu ý: kết quả train hoàn hảo không có nghĩa là mô hình tốt nhất.
- Decision Tree không giới hạn độ sâu có nguy cơ overfitting cao hơn, đặc biệt khi áp dụng trên dữ liệu mới ngoài thực tế.

### Random Forest

Kết quả:

- `Train F1 = 0.9998`
- `Test F1 = 0.9859`
- `Test ROC-AUC = 0.9987`

Nhận xét:

- Random Forest đạt kết quả tốt nhất trên tập test.
- Mô hình vẫn có train score rất cao, nhưng test score cũng cao, nên khả năng tổng quát hóa tốt.
- So với Decision Tree đơn lẻ, Random Forest ổn định hơn vì kết hợp nhiều cây quyết định.
- Đây là mô hình nên chọn nếu ưu tiên hiệu năng dự đoán.

---

## 4. Nhận xét confusion matrix

Dựa trên classification report, tập test có:

- 323 hồ sơ `Rejected`
- 531 hồ sơ `Approved`

Với mô hình Random Forest:

| Thực tế | Dự đoán Rejected | Dự đoán Approved |
|---|---:|---:|
| Rejected | 314 | 9 |
| Approved | 6 | 525 |

Diễn giải:

- Mô hình dự đoán đúng 314 hồ sơ bị từ chối.
- Mô hình dự đoán đúng 525 hồ sơ được phê duyệt.
- Có 9 trường hợp False Positive: thực tế `Rejected` nhưng mô hình dự đoán `Approved`.
- Có 6 trường hợp False Negative: thực tế `Approved` nhưng mô hình dự đoán `Rejected`.

Trong bối cảnh ngân hàng:

- False Positive có thể gây rủi ro tài chính vì ngân hàng phê duyệt nhầm hồ sơ đáng lẽ bị từ chối.
- False Negative có thể làm mất khách hàng tốt vì ngân hàng từ chối nhầm hồ sơ đáng lẽ được phê duyệt.

Tùy mục tiêu nghiệp vụ, ngân hàng có thể ưu tiên:

- Precision nếu muốn giảm phê duyệt nhầm hồ sơ rủi ro.
- Recall nếu muốn giảm bỏ sót khách hàng tốt.
- F1-score nếu muốn cân bằng cả hai mục tiêu.

---

## 5. Nhận xét learning curve

Learning curve cho thấy sự thay đổi hiệu năng khi tăng dần số lượng mẫu train.

### Logistic Regression

| Tỉ lệ train | Test Accuracy | Test F1 | Test ROC-AUC |
|---:|---:|---:|---:|
| 10% | 0.9075 | 0.9271 | 0.9698 |
| 50% | 0.9192 | 0.9356 | 0.9729 |
| 100% | 0.9145 | 0.9321 | 0.9726 |

Nhận xét:

- Kết quả khá ổn định khi tăng dữ liệu.
- Train và test gần nhau, cho thấy Logistic Regression ít overfitting.
- Tuy nhiên, F1-score không tăng mạnh khi thêm dữ liệu, cho thấy mô hình có thể bị giới hạn bởi giả định tuyến tính.

### Decision Tree

| Tỉ lệ train | Test Accuracy | Test F1 | Test ROC-AUC |
|---:|---:|---:|---:|
| 10% | 0.9625 | 0.9700 | 0.9583 |
| 50% | 0.9719 | 0.9771 | 0.9886 |
| 100% | 0.9684 | 0.9744 | 0.9884 |

Nhận xét:

- Decision Tree đạt kết quả tốt ngay cả khi dùng ít dữ liệu.
- Khi tăng dữ liệu, Test F1 cải thiện rồi dao động nhẹ.
- Mô hình có train score cao hơn test score, nhưng với `max_depth=5`, mức overfitting chưa nghiêm trọng.

### Decision Tree No Limit

| Tỉ lệ train | Train F1 | Test F1 | Test ROC-AUC |
|---:|---:|---:|---:|
| 10% | 1.0000 | 0.9700 | 0.9583 |
| 50% | 1.0000 | 0.9821 | 0.9767 |
| 100% | 1.0000 | 0.9831 | 0.9776 |

Nhận xét:

- Train F1 luôn bằng 1.0000, nghĩa là mô hình học gần như hoàn hảo trên tập train.
- Test F1 cao nhưng vẫn thấp hơn train, cho thấy có dấu hiệu overfitting.
- Đây là ví dụ tốt để sinh viên thấy rằng mô hình học thuộc train không nhất thiết là mô hình đáng tin cậy nhất.

### Random Forest

| Tỉ lệ train | Test Accuracy | Test F1 | Test ROC-AUC |
|---:|---:|---:|---:|
| 10% | 0.9660 | 0.9729 | 0.9962 |
| 50% | 0.9778 | 0.9822 | 0.9977 |
| 100% | 0.9824 | 0.9859 | 0.9987 |

Nhận xét:

- Random Forest cải thiện khá rõ khi tăng dữ liệu.
- Test F1 tăng từ 0.9729 lên 0.9859.
- Test ROC-AUC luôn rất cao, cho thấy khả năng phân biệt hai lớp rất tốt.
- Đây là mô hình tận dụng thêm dữ liệu tốt nhất trong các mô hình đã thử.

---

## 6. Nhận xét validation curve của Decision Tree

Validation curve được vẽ bằng cách thay đổi `max_depth`.

| max_depth | Train F1 | Test F1 | Test ROC-AUC |
|---:|---:|---:|---:|
| 1 | 0.9585 | 0.9731 | 0.9718 |
| 3 | 0.9691 | 0.9771 | 0.9955 |
| 5 | 0.9790 | 0.9744 | 0.9884 |
| 7 | 0.9872 | 0.9777 | 0.9850 |
| 10 | 0.9960 | 0.9832 | 0.9849 |
| 15 | 1.0000 | 0.9831 | 0.9776 |
| None | 1.0000 | 0.9831 | 0.9776 |

Nhận xét:

- Khi `max_depth` tăng, Train F1 tăng dần và đạt 1.0000 từ `max_depth=15`.
- Test F1 tốt nhất ở khoảng `max_depth=10`, đạt khoảng 0.9832.
- Khi cây quá sâu, Train F1 tiếp tục hoàn hảo nhưng ROC-AUC trên test giảm, cho thấy mô hình bắt đầu học quá chi tiết dữ liệu train.
- Với Decision Tree, `max_depth` nên được giới hạn thay vì để `None`.

---

## 7. Kết luận cuối bài

Trong bài toán Loan Approval Prediction, **Random Forest** là mô hình phù hợp nhất trong các mô hình đã thử.

Lý do:

- Có `Test F1` cao nhất: 0.9859.
- Có `Test ROC-AUC` cao nhất: 0.9987.
- Confusion matrix cho thấy số lỗi thấp: 9 False Positive và 6 False Negative.
- Learning curve cho thấy mô hình cải thiện khi tăng dữ liệu train.

Kết luận sư phạm:

- Logistic Regression là baseline tốt, ổn định và ít overfitting.
- Decision Tree dễ hiểu, phù hợp để minh họa cách phân loại bằng luật điều kiện.
- Decision Tree không giới hạn giúp minh họa overfitting.
- Random Forest cho hiệu năng tốt nhất nhờ kết hợp nhiều cây.

Khi triển khai thực tế, không nên chỉ dựa vào Accuracy. Cần xem thêm Precision, Recall, F1-score, ROC-AUC và confusion matrix để đánh giá rủi ro phê duyệt nhầm hoặc từ chối nhầm hồ sơ vay.


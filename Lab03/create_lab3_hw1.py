import json
from pathlib import Path

out = Path(r'D:/Intro2DS/Lab03/24280018_PhamTienPhat_HW1.ipynb')


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    }

cells = []

cells.append(md("""
# LAB 03 - HOMEWORK

## Thông tin sinh viên
- MSSV: 24280018
- Họ tên: Phạm Tiến Phát

## Nội dung bài làm
Bài làm được thực hiện theo 3 vai trò:
1. **Data Engineer**: Làm sạch, chuẩn hóa và tích hợp dữ liệu để tạo ra bộ dữ liệu sạch dùng chung.
2. **Data Analyst**: Trực quan hóa dữ liệu, phân tích xu hướng và trả lời các câu hỏi kinh doanh.
3. **Data Scientist**: Xây dựng mô hình hồi quy để dự đoán `Profit` từ dữ liệu đã làm sạch.
"""))

cells.append(md("""
# 1. Data Engineer

## Mục tiêu
Xử lý các file dữ liệu đầu vào, chuẩn hóa và làm sạch dữ liệu, sau đó tạo ra một file dữ liệu sạch dùng chung là `Orders_clean`.
"""))

cells.append(code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)
"""))

cells.append(code("""
from google.colab import drive
drive.mount('/content/drive')
"""))

cells.append(code("""
working_path = '/content/drive/My Drive/HCMUS/Intro2DS/Lab/Lab03/'
"""))

cells.append(md("""
## 1.1 Đọc và tiền xử lý Orders
Dữ liệu Orders gồm 2 file `lab03_Orders_Q1.csv` và `lab03_Orders_Q2.csv`.
Cần chuẩn hóa tên cột, chọn các cột cần thiết và gộp 2 file lại thành một bảng duy nhất.
"""))

cells.append(code("""
df_orders_Q1 = pd.read_csv(working_path + 'lab03_Orders_Q1.csv')
df_orders_Q2 = pd.read_csv(working_path + 'lab03_Orders_Q2.csv')

print('Q1 shape:', df_orders_Q1.shape)
print('Q2 shape:', df_orders_Q2.shape)
"""))

cells.append(code("""
# Đổi tên cột Price -> Unit_Price để đồng nhất
df_orders_Q2 = df_orders_Q2.rename(columns={'Price': 'Unit_Price'})

common_cols = [
    'Order_ID', 'Order_Date', 'Order_Priority', 'Customer_ID',
    'Industry_name', 'Province', 'Region', 'Channel',
    'Product_ID', 'Sales', 'Quantity', 'Unit_Price',
    'Discount', 'Profit', 'Returned_Reason'
]

df_orders_Q1 = df_orders_Q1[common_cols]
df_orders_Q2 = df_orders_Q2[common_cols]

df_orders = pd.concat([df_orders_Q1, df_orders_Q2], ignore_index=True)
print('Orders shape after concat:', df_orders.shape)
df_orders.head()
"""))

cells.append(code("""
for col in ['Order_ID', 'Order_Priority', 'Customer_ID', 'Industry_name',
            'Province', 'Region', 'Channel', 'Product_ID', 'Returned_Reason']:
    df_orders[col] = df_orders[col].astype(str).str.strip()

df_orders['Returned_Reason'] = df_orders['Returned_Reason'].replace(['nan', 'NaN', '\"\"', ''], np.nan)

df_orders['Industry_name'] = df_orders['Industry_name'].replace({
    'Điện lạnh': 'Hàng điện lạnh',
    'Điện tử': 'Hàng điện tử',
    'Thiết bị gia dụng': 'Hàng điện lạnh',
    'Thiết bị điện tử': 'Hàng điện tử'
})

df_orders['Province'] = df_orders['Province'].replace({
    'TP HCM': 'Hồ Chí Minh',
    'Hồ Chí Minh ': 'Hồ Chí Minh',
    'Tây Ninh ': 'Tây Ninh',
    'Hà Tỉnh': 'Hà Tĩnh'
})

num_cols = ['Sales', 'Quantity', 'Unit_Price', 'Discount', 'Profit']
for col in num_cols:
    df_orders[col] = pd.to_numeric(df_orders[col], errors='coerce')

df_orders['Order_Date'] = pd.to_datetime(df_orders['Order_Date'], errors='coerce')

print('Số dòng bị lỗi ngày:', df_orders['Order_Date'].isna().sum())
print('Số dòng thiếu Unit_Price:', df_orders['Unit_Price'].isna().sum())
print('Số dòng trùng hoàn toàn:', df_orders.duplicated().sum())
"""))

cells.append(code("""
mask_missing_price = (
    df_orders['Unit_Price'].isna() &
    df_orders['Sales'].notna() &
    df_orders['Quantity'].notna()
)
df_orders.loc[mask_missing_price, 'Unit_Price'] = (
    df_orders.loc[mask_missing_price, 'Sales'] / df_orders.loc[mask_missing_price, 'Quantity']
)

df_orders = df_orders.dropna(subset=['Order_Date'])
df_orders = df_orders.drop_duplicates()

print('Orders shape after cleaning:', df_orders.shape)
df_orders.head()
"""))

cells.append(md("""
### Nhận xét
- Dữ liệu Orders ban đầu có sự khác nhau về tên cột giữa 2 file Q1 và Q2.
- Có một số giá trị thiếu, dữ liệu ngày sai định dạng và dòng bị trùng.
- Sau khi làm sạch, dữ liệu đã sẵn sàng để join với bảng Customer và Product.
"""))

cells.append(md("""
## 1.2 Đọc và tiền xử lý Customer
"""))

cells.append(code("""
df_cus = pd.read_csv(working_path + 'lab03_CUS.csv')
print('Customer shape:', df_cus.shape)
df_cus.head()
"""))

cells.append(code("""
df_cus.columns = [c.strip().replace(' ', '_') for c in df_cus.columns]
df_cus = df_cus.rename(columns={'Phone_number': 'Phone_Number'})

for col in ['Customer_ID', 'Customer_Name', 'Customer_Segment', 'Province']:
    df_cus[col] = df_cus[col].astype(str).str.strip()

df_cus['Province'] = df_cus['Province'].replace({
    'TP HCM': 'Hồ Chí Minh',
    'Hà Tỉnh': 'Hà Tĩnh'
})

df_cus['Birthday'] = pd.to_datetime(df_cus['Birthday'], errors='coerce')
df_cus = df_cus.drop_duplicates(subset=['Customer_ID'])

df_cus.info()
"""))

cells.append(md("""
### Nhận xét
- Bảng Customer được chuẩn hóa tên cột để dễ xử lý.
- Đã chuẩn hóa tên tỉnh/thành và loại bỏ trùng theo `Customer_ID`.
"""))

cells.append(md("""
## 1.3 Đọc và tiền xử lý Product
"""))

cells.append(code("""
df_prod = pd.read_csv(working_path + 'lab03_Product.csv')
print('Product shape:', df_prod.shape)
df_prod.head()
"""))

cells.append(code("""
df_prod.columns = [c.strip().replace(' ', '_') for c in df_prod.columns]

for col in ['Product_ID', 'Product_Category', 'Product_Sub-Category', 'Product_Name']:
    df_prod[col] = df_prod[col].astype(str).str.strip()

df_prod['Product_Category'] = df_prod['Product_Category'].replace({
    'Điện lạnh': 'Hàng điện lạnh',
    'Điện tử': 'Hàng điện tử'
})

df_prod = df_prod.drop_duplicates(subset=['Product_ID'])
df_prod.info()
"""))

cells.append(md("""
### Nhận xét
- Bảng Product có một số giá trị category chưa đồng nhất.
- Sau khi chuẩn hóa, bảng Product có thể join ổn định với Orders qua `Product_ID`.
"""))

cells.append(md("""
## 1.4 Join dữ liệu và tạo Orders_clean
"""))

cells.append(code("""
df_orders_cus = df_orders.merge(
    df_cus,
    on='Customer_ID',
    how='left',
    suffixes=('_order', '_customer')
)

df_final = df_orders_cus.merge(
    df_prod,
    on='Product_ID',
    how='left'
)

print('Final shape:', df_final.shape)
df_final.head()
"""))

cells.append(code("""
df_final['Return_Flag'] = np.where(df_final['Returned_Reason'].notna(), 1, 0)
df_final['Year'] = df_final['Order_Date'].dt.year
df_final['Month'] = df_final['Order_Date'].dt.month
df_final['YearMonth'] = df_final['Order_Date'].dt.to_period('M').astype(str)
df_final['Profit_Margin'] = np.where(df_final['Sales'] != 0, df_final['Profit'] / df_final['Sales'], np.nan)

Orders_clean = df_final.copy()

print('Orders_clean shape:', Orders_clean.shape)
Orders_clean.head()
"""))

cells.append(md("""
## Kết luận phần Data Engineer
Đã thực hiện:
- Đọc và làm sạch 2 file Orders
- Đọc và làm sạch Customer
- Đọc và làm sạch Product
- Join các bảng để tạo ra bộ dữ liệu sạch `Orders_clean`
"""))

cells.append(md("""
# 2. Data Analyst

## Mục tiêu
Sử dụng `Orders_clean` để trực quan hóa dữ liệu, khám phá xu hướng và trả lời các câu hỏi kinh doanh.
"""))

cells.append(md("""
## Task 1 — Phân tích xu hướng theo thời gian
Phân tích Sales, Profit và số lượng Orders theo tháng.
"""))

cells.append(code("""
monthly = Orders_clean.groupby('YearMonth').agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum'),
    Orders=('Order_ID', 'nunique')
).reset_index()

monthly
"""))

cells.append(code("""
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

sns.lineplot(data=monthly, x='YearMonth', y='Sales', marker='o', ax=axes[0])
axes[0].set_title('Sales theo tháng')
axes[0].tick_params(axis='x', rotation=45)

sns.lineplot(data=monthly, x='YearMonth', y='Profit', marker='o', ax=axes[1], color='green')
axes[1].set_title('Profit theo tháng')
axes[1].tick_params(axis='x', rotation=45)

sns.lineplot(data=monthly, x='YearMonth', y='Orders', marker='o', ax=axes[2], color='orange')
axes[2].set_title('Số lượng Orders theo tháng')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
"""))

cells.append(md("""
### Nhận xét Task 1
- Doanh thu và lợi nhuận có sự biến động theo thời gian.
- Có những giai đoạn tăng mạnh và giảm nhẹ giữa các tháng.
- Nếu xuất hiện đỉnh doanh thu rõ rệt ở một vài tháng, có thể xem đó là dấu hiệu của yếu tố mùa vụ hoặc chương trình bán hàng.
"""))

cells.append(md("""
## Task 2 — Phân tích theo khu vực (Region / Province)
"""))

cells.append(code("""
region_summary = Orders_clean.groupby('Region').agg(
    Sales=('Sales', 'sum'),
    Profit=('Profit', 'sum')
).reset_index()

region_summary['Profit_Margin'] = region_summary['Profit'] / region_summary['Sales']
region_summary
"""))

cells.append(code("""
plt.figure(figsize=(10, 5))
sns.barplot(data=region_summary, x='Region', y='Sales')
plt.title('Sales theo Region')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=region_summary, x='Region', y='Profit')
plt.title('Profit theo Region')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=region_summary, x='Region', y='Profit_Margin')
plt.title('Profit Margin theo Region')
plt.xticks(rotation=45)
plt.show()
"""))

cells.append(code("""
top_province = Orders_clean.groupby('Province_order').agg(
    Sales=('Sales', 'sum')
).reset_index().sort_values('Sales', ascending=False).head(10)

top_province
"""))

cells.append(code("""
plt.figure(figsize=(12, 6))
sns.barplot(data=top_province, x='Sales', y='Province_order')
plt.title('Top 10 Province theo Sales')
plt.show()
"""))

cells.append(md("""
### Nhận xét Task 2
- Có thể xác định khu vực có doanh thu cao nhất dựa trên biểu đồ Sales theo Region.
- So sánh Profit và Profit Margin giúp phát hiện khu vực bán nhiều nhưng lợi nhuận thấp.
- Những tỉnh/thành đứng đầu về Sales là các địa bàn kinh doanh trọng điểm.
"""))

cells.append(md("""
## Task 3 — Phân tích theo Customer
"""))

cells.append(code("""
industry_sales = Orders_clean.groupby('Industry_name').agg(
    Sales=('Sales', 'sum')
).reset_index().sort_values('Sales', ascending=False)

industry_sales
"""))

cells.append(code("""
plt.figure(figsize=(8, 5))
sns.barplot(data=industry_sales, x='Industry_name', y='Sales')
plt.title('Sales theo Industry_name')
plt.xticks(rotation=20)
plt.show()
"""))

cells.append(code("""
customer_sales = Orders_clean.groupby('Customer_Name').agg(
    Sales=('Sales', 'sum')
).reset_index().sort_values('Sales', ascending=False)

customer_sales['Cumulative_Sales'] = customer_sales['Sales'].cumsum()
customer_sales['Cumulative_Pct'] = customer_sales['Cumulative_Sales'] / customer_sales['Sales'].sum()

customer_sales.head()
"""))

cells.append(code("""
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(customer_sales) + 1), customer_sales['Cumulative_Pct'], marker='o')
plt.axhline(0.8, color='red', linestyle='--')
plt.title('Pareto 80/20 - Customer Revenue')
plt.xlabel('Số khách hàng')
plt.ylabel('Tỷ lệ doanh thu tích lũy')
plt.show()

num_top_customers_80 = (customer_sales['Cumulative_Pct'] <= 0.8).sum()
print('Số khách hàng đóng góp khoảng 80% doanh thu:', num_top_customers_80)
"""))

cells.append(code("""
heatmap_data = Orders_clean.pivot_table(
    index='Industry_name',
    columns='Region',
    values='Sales',
    aggfunc='sum',
    fill_value=0
)

plt.figure(figsize=(10, 5))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlGnBu')
plt.title('Industry vs Region (Sales)')
plt.show()
"""))

cells.append(md("""
### Nhận xét Task 3
- Nhóm khách hàng chính là nhóm có doanh thu cao nhất theo `Industry_name`.
- Một số ít khách hàng có thể đóng góp phần lớn doanh thu theo quy luật Pareto 80/20.
- Heatmap giúp nhận biết ngành hàng nào mạnh ở từng khu vực.
"""))

cells.append(md("""
## Task 4 — Phân tích Returned_Reason
"""))

cells.append(code("""
return_by_product = Orders_clean.groupby('Product_Category').agg(
    Return_Rate=('Return_Flag', 'mean')
).reset_index()

return_by_region = Orders_clean.groupby('Region').agg(
    Return_Rate=('Return_Flag', 'mean')
).reset_index()

return_by_channel = Orders_clean.groupby('Channel').agg(
    Return_Rate=('Return_Flag', 'mean')
).reset_index()
"""))

cells.append(code("""
plt.figure(figsize=(8, 5))
sns.barplot(data=return_by_product, x='Product_Category', y='Return_Rate')
plt.title('Tỷ lệ trả hàng theo Product Category')
plt.xticks(rotation=20)
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(data=return_by_region, x='Region', y='Return_Rate')
plt.title('Tỷ lệ trả hàng theo Region')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=return_by_channel, x='Channel', y='Return_Rate')
plt.title('Tỷ lệ trả hàng theo Channel')
plt.show()
"""))

cells.append(code("""
returned_reason_dist = Orders_clean['Returned_Reason'].fillna('No Return').value_counts()
returned_reason_dist
"""))

cells.append(code("""
plt.figure(figsize=(10, 5))
returned_reason_dist.plot(kind='bar')
plt.title('Phân bố Returned_Reason')
plt.xticks(rotation=45)
plt.show()
"""))

cells.append(md("""
### Nhận xét Task 4
- Có thể xác định khu vực, nhóm sản phẩm hoặc kênh bán có tỷ lệ trả hàng cao nhất.
- Nếu lý do trả hàng tập trung vào một số nguyên nhân cụ thể thì doanh nghiệp có thể ưu tiên xử lý các nguyên nhân đó.
- Nếu dữ liệu `Returned_Reason` còn thiếu nhiều, cần lưu ý rằng kết luận chỉ mang tính tham khảo.
"""))

cells.append(md("""
## Kết luận phần Data Analyst
Qua trực quan hóa dữ liệu:
- Đã quan sát được xu hướng doanh thu, lợi nhuận và số đơn theo thời gian
- Đã xác định các khu vực và tỉnh/thành quan trọng
- Đã phân tích nhóm khách hàng chính
- Đã xem xét tình hình trả hàng theo nhiều góc độ
"""))

cells.append(md("""
# 3. Data Scientist

## Mục tiêu
Xây dựng mô hình Regression để dự đoán `Profit` dựa trên dữ liệu đã làm sạch.
"""))

cells.append(md("""
## Task 1 — Chuẩn bị dữ liệu cho mô hình
Chọn biến đầu vào phù hợp, xử lý dữ liệu số và dữ liệu phân loại.
"""))

cells.append(code("""
model_df = Orders_clean.copy()

features = [
    'Order_Priority', 'Industry_name', 'Province_order', 'Region', 'Channel',
    'Quantity', 'Unit_Price', 'Discount', 'Sales',
    'Customer_Segment', 'Product_Category', 'Product_Sub-Category'
]

target = 'Profit'

model_df = model_df[features + [target]].dropna(subset=[target])

X = model_df[features]
y = model_df[target]

categorical_cols = [
    'Order_Priority', 'Industry_name', 'Province_order', 'Region', 'Channel',
    'Customer_Segment', 'Product_Category', 'Product_Sub-Category'
]

numeric_cols = ['Quantity', 'Unit_Price', 'Discount', 'Sales']

print('X shape:', X.shape)
print('y shape:', y.shape)
"""))

cells.append(md("""
## Task 2 — Chia train/test
"""))

cells.append(code("""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print('Train size:', X_train.shape)
print('Test size:', X_test.shape)
"""))

cells.append(md("""
## Task 3 — Huấn luyện Linear Regression
"""))

cells.append(code("""
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
"""))

cells.append(md("""
## Task 4 — Đánh giá mô hình
"""))

cells.append(code("""
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('MAE:', mae)
print('RMSE:', rmse)
print('R2:', r2)
"""))

cells.append(md("""
### Nhận xét Task 4
- **MAE** cho biết sai số tuyệt đối trung bình của mô hình.
- **RMSE** cho biết mức sai số, nhạy hơn với các dự đoán sai lớn.
- **R²** cho biết mức độ giải thích biến động của `Profit` từ các biến đầu vào.
"""))

cells.append(md("""
## Task 5 — Feature Importance
"""))

cells.append(code("""
feature_names_num = numeric_cols
feature_names_cat = model.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .named_steps['onehot'] \
    .get_feature_names_out(categorical_cols)

feature_names = np.concatenate([feature_names_num, feature_names_cat])
coefficients = model.named_steps['regressor'].coef_

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)

coef_df.head(15)
"""))

cells.append(code("""
plt.figure(figsize=(12, 6))
sns.barplot(data=coef_df.head(15), x='Coefficient', y='Feature')
plt.title('Top 15 Feature Importance (Linear Regression)')
plt.show()
"""))

cells.append(md("""
### Nhận xét Task 5
- Các biến có hệ số lớn nhất là những biến ảnh hưởng mạnh nhất đến `Profit`.
- Hệ số dương cho thấy biến đó làm tăng lợi nhuận.
- Hệ số âm cho thấy biến đó có xu hướng làm giảm lợi nhuận.
"""))

cells.append(md("""
## Kết luận phần Data Scientist
- Đã xây dựng mô hình `Linear Regression` để dự đoán `Profit`
- Đã đánh giá mô hình bằng các chỉ số `MAE`, `RMSE`, `R²`
- Đã phân tích mức độ ảnh hưởng của các biến đầu vào đến lợi nhuận
"""))

cells.append(md("""
# 4. Kết luận chung

Trong bài lab này, em đã thực hiện đầy đủ 3 vai trò:

### 1. Data Engineer
- Làm sạch và chuẩn hóa dữ liệu từ nhiều nguồn
- Gộp dữ liệu Orders
- Join với Customer và Product
- Tạo ra dataset `Orders_clean`

### 2. Data Analyst
- Phân tích xu hướng doanh thu, lợi nhuận, số đơn hàng
- Phân tích theo khu vực, khách hàng, ngành hàng
- Xem xét tình trạng trả hàng và lý do trả hàng

### 3. Data Scientist
- Xây dựng mô hình hồi quy dự đoán `Profit`
- Đánh giá mô hình bằng các metric phù hợp
- Xác định các biến ảnh hưởng nhiều đến lợi nhuận

Qua đó, bài làm đã đáp ứng yêu cầu của Homework ở cả 3 vai trò.
"""))

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.x'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 5
}

with out.open('w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(out)
print('cells:', len(cells))

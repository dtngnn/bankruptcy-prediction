# Dự Đoán Phá Sản Doanh Nghiệp Bằng Học Máy

## Mục Lục

- [Giới thiệu](#giới-thiệu)
- [Mô tả Bộ Dữ Liệu](#mô-tả-bộ-dữ-liệu)
- [Cấu trúc Project](#cấu-trúc-project)
- [Cài đặt và Yêu cầu](#cài-đặt-và-yêu-cầu)
- [Quy trình Thực hiện](#quy-trình-thực-hiện)
  - [Bước 1: Khám phá Dữ liệu (EDA)](#bước-1-khám-phá-dữ-liệu-eda)
  - [Bước 2: Tiền xử lý Dữ liệu](#bước-2-tiền-xử-lý-dữ-liệu)
  - [Bước 3: Huấn luyện Mô hình](#bước-3-huấn-luyện-mô-hình)
  - [Bước 4: Đánh giá và So sánh](#bước-4-đánh-giá-và-so-sánh)
- [Kết quả](#kết-quả)
- [Kết luận](#kết-luận)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Giới thiệu

Dự án xây dựng mô hình học máy để dự đoán nguy cơ phá sản của doanh nghiệp dựa trên các chỉ số tài chính. Đây là bài toán **phân loại nhị phân trên dữ liệu mất cân bằng cực đoan** (lớp phá sản chỉ chiếm 0.6% tổng thể), thuộc dạng *rare-event prediction* — rất phổ biến trong lĩnh vực tài chính.

**Mục tiêu:**
- Xác định liệu một doanh nghiệp có nguy cơ rơi vào tình trạng phá sản hay không.
- So sánh hiệu suất của nhiều thuật toán phân loại khác nhau.
- Tìm ra mô hình tối ưu nhất cho bài toán dữ liệu mất cân bằng.
- Xác định các chỉ số tài chính quan trọng nhất ảnh hưởng đến dự đoán phá sản.

---

## Mô tả Bộ Dữ Liệu

Bộ dữ liệu `bankruptcy_data.xlsx` gồm **92.872 quan sát** với **12 đặc trưng tài chính** và **1 biến mục tiêu**.

### Các đặc trưng (Features)

| Đặc trưng | Mô tả |
|:---|:---|
| **EPS** | Lợi nhuận trên mỗi cổ phiếu (Earnings Per Share) |
| **Liquidity** | Khả năng thanh khoản — chuyển đổi tài sản thành tiền mặt |
| **Profitability** | Khả năng sinh lời trong hoạt động kinh doanh |
| **Productivity** | Năng suất hoạt động — hiệu quả tạo ra giá trị |
| **Leverage Ratio** | Tỷ lệ đòn bẩy tài chính (nợ / vốn chủ sở hữu) |
| **Asset Turnover** | Vòng quay tài sản — hiệu quả sử dụng tài sản |
| **Operational Margin** | Biên lợi nhuận hoạt động |
| **Return on Equity (ROE)** | Lợi nhuận trên vốn chủ sở hữu |
| **Market Book Ratio** | Tỷ giá thị trường / giá sổ sách |
| **Assets Growth** | Tốc độ tăng trưởng tài sản |
| **Sales Growth** | Tốc độ tăng trưởng doanh thu |
| **Employee Growth** | Tốc độ tăng trưởng nhân sự |

### Biến mục tiêu

| Giá trị | Ý nghĩa |
|:---|:---|
| `BK = 0` | Doanh nghiệp **không phá sản** (99.4%) |
| `BK = 1` | Doanh nghiệp **phá sản** (0.6% — 558 mẫu) |

> **Lưu ý:** Tỷ lệ mất cân bằng cực đoan (0.6%) là thách thức lớn nhất của bài toán. Nếu không xử lý, mô hình có thể đoán toàn bộ là "không phá sản" và vẫn đạt Accuracy ~99%.

---

## Cấu trúc Project

```
project_python/
├── project.ipynb          # Notebook chính chứa toàn bộ phân tích
├── bankruptcy_data.xlsx   # Dữ liệu tài chính đầu vào
└── README.md              # Tài liệu mô tả dự án
```

---

## Cài đặt và Yêu cầu

### Ngôn ngữ & Môi trường
- Python 3.13+
- Jupyter Notebook / VS Code

### Thư viện cần thiết

```bash
pip install numpy pandas scipy matplotlib seaborn missingno
pip install scikit-learn imbalanced-learn xgboost
```

| Thư viện | Mục đích |
|:---|:---|
| `numpy`, `pandas` | Xử lý và thao tác dữ liệu |
| `scipy` | Tính toán thống kê, Winsorization |
| `matplotlib`, `seaborn` | Trực quan hóa dữ liệu |
| `missingno` | Phân tích mẫu dữ liệu bị thiếu |
| `scikit-learn` | Mô hình ML, tiền xử lý, đánh giá |
| `imbalanced-learn` | Xử lý dữ liệu mất cân bằng (SMOTE, ENN, ImbPipeline) |
| `xgboost` | Thuật toán Gradient Boosting nâng cao |

---

## Quy trình Thực hiện

### Bước 1: Khám phá Dữ liệu (EDA)

#### 1.1. Kiểu dữ liệu
- Toàn bộ đặc trưng đều là kiểu `float64`, biến mục tiêu `BK` là `int64` nhị phân.
- Dữ liệu đã ở dạng sẵn sàng cho Machine Learning, không cần mã hóa (encoding).

#### 1.2. Phân phối và Thống kê mô tả
- Hầu hết các biến có **phân phối lệch** (skewed) với đuôi dày, không tuân theo phân phối chuẩn.
- Khoảng cách giữa Min–Max so với IQR rất lớn, cho thấy sự biến động mạnh — đặc trưng của dữ liệu tài chính.

#### 1.3. Phân tích Mất cân bằng
- Lớp phá sản (`BK = 1`) chỉ chiếm **0.6%** (558 / 92.872 mẫu).
- Đây là bài toán *rare-event*, cần các kỹ thuật đặc biệt để xử lý.

#### 1.4. Phát hiện Ngoại lai (Outliers)
- Outlier chiếm **12–17%** tổng thể dữ liệu trên hầu hết các biến.
- Nhiều outlier chính là **đặc trưng của doanh nghiệp phá sản** (ROE âm, nợ cao, lợi nhuận cực thấp) → không thể xóa.
- **Quyết định:** Giữ lại toàn bộ quan sát, sử dụng Winsorization để giảm ảnh hưởng cực đoan.

#### 1.5. Tương quan giữa các biến
- Không có cặp biến nào có tương quan cao (> 0.7), không có hiện tượng đa cộng tuyến nghiêm trọng.
- Giữ lại toàn bộ 12 đặc trưng để mô hình khai thác tối đa thông tin.

#### 1.6. Giá trị trùng lặp
- Phát hiện và loại bỏ các bản ghi trùng lặp; **558 mẫu phá sản** vẫn được giữ nguyên.

#### 1.7. Phân tích Dữ liệu bị thiếu (Missing Values)
- Nhiều cột bị thiếu dữ liệu (tỷ lệ < 20%), đặc biệt: `Employee Growth`, `Assets Growth`, `Sales Growth`, `Operational Margin`.
- Phát hiện mẫu thiếu có tương quan cao giữa các nhóm biến:
  - `Asset Turnover`, `Profitability`, `Productivity`, `Liquidity` — 100% trùng nhau.
  - `Assets Growth` và `Sales Growth` — 100% trùng nhau.
  - `ROE` và `EPS` — tương quan cao.
- **Phương pháp xử lý:** KNN Imputer (phù hợp khi biến có tương quan cao).

---

### Bước 2: Tiền xử lý Dữ liệu

Quy trình tiền xử lý được thiết kế **tuần tự và nghiêm ngặt** để tránh rò rỉ dữ liệu (data leakage).

#### 2.1. Chia tập Train / Test

```
Train : Test = 75% : 25% (random_state=42)
```

> Tách dữ liệu **trước khi** thực hiện bất kỳ bước xử lý nào, đảm bảo tập test hoàn toàn "chưa thấy" trong quá trình huấn luyện.

#### 2.2. Điền giá trị thiếu — KNN Imputer

- Sử dụng `KNNImputer` với `n_neighbors=5`, `weights='distance'`, `metric='nan_euclidean'`.
- **Chỉ fit trên tập train**, sau đó transform cả train và test.
- **Lý do chọn KNN Imputer:** Dữ liệu tài chính có nhiều biến tương quan, giá trị thiếu có thể suy luận chính xác từ các biến lân cận.

#### 2.3. Xử lý Ngoại lai — Winsorization (5% – 95%)

- Các giá trị nằm dưới phân vị 5% hoặc trên phân vị 95% được thay thế bằng giá trị biên tương ứng.
- **Giới hạn Winsor được tính từ tập train**, áp dụng cho cả train và test.
- **Ưu điểm:** Giảm ảnh hưởng của cực ngoại lai mà không xóa dữ liệu, bảo toàn kích thước mẫu.

#### 2.4. Xử lý Mất cân bằng — SMOTE-ENN

- **SMOTE** (Synthetic Minority Oversampling Technique): Tạo mẫu tổng hợp cho lớp thiểu số bằng cách nội suy giữa các mẫu lân cận (`k_neighbors=5`).
- **ENN** (Edited Nearest Neighbours): Loại bỏ các mẫu tổng hợp bị nhiễu hoặc quá sát ranh giới quyết định (`n_neighbors=3`).
- **Kết hợp SMOTE-ENN:** Trước tiên SMOTE tăng lượng mẫu thiểu số, sau đó ENN dọn dẹp → cho ra tập dữ liệu cân bằng và sạch hơn.
- **Chỉ áp dụng trên tập train**, tập test giữ nguyên phân phối gốc.

#### 2.5. Chuẩn hóa — RobustScaler

- Sử dụng `RobustScaler` (dựa trên median và IQR) thay vì StandardScaler.
- **Lý do:** Dữ liệu tài chính vẫn còn ngoại lai sau Winsorization; RobustScaler ít bị ảnh hưởng bởi outlier hơn so với StandardScaler.

#### Sơ đồ Pipeline tổng thể

```
Dữ liệu gốc
    │
    ├── Train/Test Split (75/25, stratified)
    │
    ├── [TRAIN PATH]
    │       ├── KNN Imputer (fit + transform)
    │       ├── Winsorization (tính giới hạn + áp dụng)
    │       ├── SMOTE-ENN (cân bằng lớp) ─── [Cho Voting Classifier]
    │       └── RobustScaler (fit + transform)
    │
    └── [TEST PATH]
            ├── KNN Imputer (transform only)
            ├── Winsorization (áp dụng giới hạn từ train)
            └── RobustScaler (transform only)
```

---

### Bước 3: Huấn luyện Mô hình

#### 3.1. Logistic Regression (Baseline)

- **Vai trò:** Mô hình chuẩn (benchmark) để so sánh — đơn giản, nhanh, dễ giải thích.
- **Cách hoạt động:** Dự đoán xác suất phá sản bằng hàm sigmoid.
- **Phân tích hệ số (Coef):** Cho phép xác định mức độ và chiều hướng ảnh hưởng của từng đặc trưng đến xác suất phá sản.

#### 3.2. Random Forest

- **Vai trò:** Đánh giá khả năng xử lý quan hệ phi tuyến.
- **Cách hoạt động:** Tập hợp nhiều cây quyết định, kết hợp kết quả bằng biểu quyết.
- **Tham số:** `n_estimators=10`, `max_features='sqrt'`, `random_state=21`.

#### 3.3. Gaussian Naive Bayes

- **Vai trò:** Kiểm tra hiệu suất với thuật toán dựa trên xác suất điều kiện.
- **Cách hoạt động:** Giả định các đặc trưng độc lập và tuân theo phân phối chuẩn; áp dụng định lý Bayes.

#### 3.4. Voting Classifier (Soft Voting)

- **Vai trò:** Kết hợp sức mạnh của 3 mô hình đơn lẻ.
- **Cách hoạt động:** Lấy trung bình xác suất dự đoán từ Logistic Regression, Random Forest, và Naive Bayes (*soft voting*) để đưa ra quyết định cuối cùng.
- **Đánh giá ổn định:** Sử dụng **Stratified K-Fold Cross-Validation (K=5)** với `ImbPipeline` để đảm bảo:
  - SMOTE-ENN chỉ được áp dụng **bên trong mỗi fold**, không rò rỉ dữ liệu.
  - Tỷ lệ lớp mục tiêu giữ nguyên trong mỗi fold.

#### 3.5. XGBoost (Đề xuất nâng cao)

- **Vai trò:** Giải pháp tối ưu cho dữ liệu mất cân bằng.
- **Cách hoạt động:** Gradient Boosting — xây dựng cây tuần tự, mỗi cây tập trung sửa lỗi của cây trước.
- **Xử lý mất cân bằng:** Sử dụng tham số `scale_pos_weight = ratio` (tỷ lệ mẫu âm / mẫu dương) thay vì SMOTE:
  - Tăng trọng số cho lớp thiểu số trong hàm loss.
  - Huấn luyện trực tiếp trên dữ liệu gốc (không qua SMOTE), tránh tạo mẫu tổng hợp.
- **Tham số:** `n_estimators=100`, `learning_rate=0.05`, `max_depth=5`.
- **Scaler riêng:** `RobustScaler` fit trên dữ liệu gốc (`train_t`), không phải dữ liệu đã resample.

---

### Bước 4: Đánh giá và So sánh

#### Các chỉ số đánh giá

| Chỉ số | Ý nghĩa | Tầm quan trọng |
|:---|:---|:---|
| **Accuracy** | Tỷ lệ dự đoán đúng tổng thể | Dễ gây hiểu lầm trên dữ liệu mất cân bằng |
| **Precision** | Trong những dự đoán "phá sản", bao nhiêu là đúng? | Giảm báo động sai |
| **Recall** | Trong tất cả ca phá sản thực, bắt được bao nhiêu? | **Ưu tiên cao** — không bỏ sót rủi ro |
| **F1 Score** | Trung bình điều hòa của Precision và Recall | Cân bằng hai chỉ số |
| **AUC** | Diện tích dưới đường cong ROC | Khả năng phân biệt giữa hai lớp |
| **Specificity** | Tỷ lệ nhận đúng doanh nghiệp lành mạnh | Tránh làm phiền doanh nghiệp tốt |
| **MCC** | Matthews Correlation Coefficient (-1 đến 1) | **Chỉ số khách quan nhất** cho dữ liệu mất cân bằng |

#### Phương pháp đánh giá nâng cao
- **Confusion Matrix:** Ma trận nhầm lẫn trực quan hóa TP, TN, FP, FN.
- **ROC Curve:** So sánh đường cong ROC giữa các mô hình.
- **Precision-Recall Curve:** Đánh giá hiệu suất trên dữ liệu mất cân bằng (quan trọng hơn ROC).
- **Feature Importance:** Xác định các chỉ số tài chính ảnh hưởng nhất đến dự đoán.

---

## Kết quả

### Bảng So sánh Hiệu suất Tổng thể

| Mô hình | Accuracy | Precision | Recall | F1 Score | AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 85.99% | 0.0331 | 0.7285 | 0.0633 | 0.8747 |
| Random Forest | 98.32% | 0.1033 | 0.2053 | 0.1375 | 0.8215 |
| Naive Bayes | 82.44% | 0.0281 | 0.7748 | 0.0543 | 0.8641 |
| Voting Classifier | 89.14% | 0.0396 | 0.6755 | 0.0749 | 0.8953 |
| **XGBoost** | **91.78%** | **0.0525** | **0.6821** | **0.0975** | **0.9075** |

### So sánh Chi tiết: XGBoost vs. Voting Classifier

| Chỉ số | XGBoost | Voting Classifier | Cải thiện |
|:---|:---:|:---:|:---:|
| Accuracy | 91.78% | 89.14% | +2.64% |
| Precision | 0.0525 | 0.0396 | +32.6% |
| Recall | 0.6821 | 0.6755 | +1.0% |
| F1 Score | 0.0975 | 0.0749 | +30.2% |
| Specificity | 0.9194 | 0.8928 | +3.0% |
| AUC | 0.9075 | 0.8953 | +1.4% |
| MCC | 0.1738 | 0.1455 | +19.5% |

### Top 10 Feature Importance (XGBoost)

| Hạng | Đặc trưng | Trọng số | Ý nghĩa |
|:---:|:---|:---:|:---|
| 1 | **Return on Equity** | 0.3880 | Hiệu quả sử dụng vốn — chỉ số dẫn đầu |
| 2 | Operational Margin | 0.0997 | Lợi nhuận hoạt động cốt lõi |
| 3 | Assets Growth | 0.0702 | Tốc độ tăng trưởng tài sản |
| 4 | Market Book Ratio | 0.0601 | Kỳ vọng thị trường vs. giá trị sổ sách |
| 5 | Leverage Ratio | 0.0573 | Mức độ sử dụng nợ |
| 6 | Asset Turnover | 0.0524 | Hiệu quả sử dụng tài sản |
| 7 | Sales Growth | 0.0504 | Tốc độ tăng trưởng doanh thu |
| 8 | Liquidity | 0.0499 | Khả năng thanh khoản |
| 9 | EPS | 0.0451 | Lợi nhuận trên mỗi cổ phiếu |
| 10 | Profitability | 0.0441 | Khả năng sinh lời tổng thể |

> **Nhận xét:** ROE chiếm gần 39% trọng số, cho thấy đây là chỉ số quyết định nhất trong dự đoán phá sản. Doanh nghiệp có ROE âm hoặc sụt giảm nghiêm trọng là dấu hiệu sớm nhất cho thấy đang "ăn vào" vốn chủ sở hữu.

### Đánh giá Tính ổn định — Stratified K-Fold CV (K=5)

Kết quả Cross-validation cho Voting Classifier (với ImbPipeline trên dữ liệu gốc):

| Chỉ số | Trung bình | Độ lệch chuẩn |
|:---|:---:|:---:|
| AUC | 0.8928 | ±0.0152 |
| F1 Score | 0.0725 | ±0.0043 |

> Độ lệch chuẩn cực thấp chứng minh mô hình hoạt động nhất quán và không bị overfitting.

---

## Kết luận

1. **XGBoost là mô hình tối ưu nhất** cho bài toán dự đoán phá sản trên dữ liệu mất cân bằng cực đoan. Mô hình đạt AUC = 0.9075 (mức xuất sắc), vượt trội so với Voting Classifier và tất cả các mô hình đơn lẻ.

2. **Phương pháp `scale_pos_weight` hiệu quả hơn SMOTE** cho XGBoost: huấn luyện trực tiếp trên dữ liệu gốc với trọng số lớp, tránh tạo mẫu tổng hợp và giảm nguy cơ rò rỉ dữ liệu.

3. **ROE là chỉ số tài chính quan trọng nhất** (38.8% trọng số), theo sau là Operational Margin và Assets Growth. Kết quả này phù hợp với lý thuyết tài chính: doanh nghiệp phá sản thường do kinh doanh không có lãi kết hợp nợ quá cao.

4. **MCC = 0.1738** — chỉ số dương ổn định trong bối cảnh dữ liệu mất cân bằng cực đoan, cho thấy mô hình vượt xa hiệu suất của bộ phân loại ngẫu nhiên.

5. **Specificity = 91.94%** — mô hình nhận đúng 92% doanh nghiệp lành mạnh, giảm thiểu báo động sai trong thực tế.

### Hạn chế và Hướng phát triển

- Precision và F1 Score vẫn ở mức thấp do đặc thù dữ liệu cực kỳ mất cân bằng (0.6%).
- Hướng nghiên cứu tiếp theo: tối ưu siêu tham số (GridSearchCV / Bayesian Optimization), thử nghiệm LightGBM / CatBoost, kết hợp chọn lọc đặc trưng nâng cao, và tối ưu ngưỡng quyết định (decision threshold) dựa trên chi phí business.

---

## Tài liệu tham khảo

1. [DATA-SCIENCE-FOR-FINANCE / Bankruptcy Prediction — serkannpolatt](https://github.com/serkannpolatt/DATA-SCIENCE-FOR-FINANCE/tree/main/Bankruptcy%20Prediction)
2. [Bankruptcy Prediction Using Machine Learning Techniques — MDPI](https://www.mdpi.com/1911-8074/15/1/35)
3. [Bankruptcy Prediction Using Machine Learning Models — IJSREM](https://ijsrem.com/download/predicting-bankruptcy-with-machine-learning-models/)
4. [Ứng dụng học máy trong dự báo rủi ro phá sản — KTPT NEU](https://ktpt.neu.edu.vn/Uploads/Bai%20bao/2023/So%20310/380842.pdf)
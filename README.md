# 📊 Phân Tích Dữ Liệu Bán Lẻ

## 📌 Giới Thiệu

Dự án phân tích dữ liệu bán lẻ từ tập dữ liệu **Retail Analysis** – mô phỏng hoạt động kinh doanh của một công ty bán lẻ tại nhiều quốc gia với gần **300.000 giao dịch**. Dự án trả lời câu hỏi nghiên cứu trọng tâm:

> **"Làm thế nào để tăng doanh thu trong bối cảnh khách hàng có thể thay đổi thói quen mua sắm?"**

---

## 📂 Dữ Liệu

**Nguồn:** [Retail Analysis on Large Dataset – Kaggle](https://www.kaggle.com/datasets/sahilprajapati143/retail-analysis-large-dataset?select=new_retail_data.csv)

**Các trường chính:**

| Cột | Mô tả |
|-----|-------|
| `Transaction_ID` | Mã định danh giao dịch |
| `Customer_ID` | Mã định danh khách hàng |
| `Age` / `Gender` / `Income` | Thông tin nhân khẩu học |
| `Customer_Segment` | Phân khúc khách hàng |
| `Date` / `Month` / `Year` | Thời gian giao dịch |
| `Product_Category` / `Product_Brand` / `Product_Type` | Thông tin sản phẩm |
| `Amount` / `Total_Amount` / `Total_Purchases` | Giá trị giao dịch |
| `Payment_Method` | Phương thức thanh toán |
| `Shipping_Method` | Phương thức vận chuyển |
| `Order_Status` | Trạng thái đơn hàng |
| `Ratings` / `Feedback` | Đánh giá khách hàng |

---

## 🔧 Tiền Xử Lý Dữ Liệu

- Chuẩn hóa kiểu dữ liệu cho các cột số và ngày tháng
- Chuẩn hóa cột `State` về định dạng thống nhất
- Điền dữ liệu thiếu cho các cột địa lý (`City`, `State`, `Country`, `Zipcode`) dựa trên thông tin liên quan
- Điền dữ liệu thiếu cho các cột giao dịch (`Amount`, `Total_Amount`, `Total_Purchases`)
- Điền dữ liệu thiếu cho các cột sản phẩm (`Product_Brand`, `Product_Category`) dựa vào cột `Product`
- Tái tạo cột `Month` và `Year` từ cột `Date`
- Xóa các cột không cần thiết và loại bỏ các giá trị không thể xử lý
- Kiểm tra và xử lý dữ liệu trùng lặp

---

## 🔍 Nội Dung Phân Tích (EDA)

### 4.1 Tổng Quan Dữ Liệu
- Kiểm tra kích thước, kiểu dữ liệu, số giá trị duy nhất và thống kê mô tả cơ bản
- Phát hiện outliers qua biểu đồ boxplot các biến số (`Age`, `Amount`, `Total_Amount`, `Total_Purchases`)

### 4.2 Phân Tích Khách Hàng
- **Nhân khẩu học:** phân bố độ tuổi, giới tính, thu nhập và phân khúc khách hàng
- **Địa lý:** USA dẫn đầu với ~128.85 triệu doanh thu; tiếp theo là UK (~86.07M), Germany (~71.64M), Australia và Canada (~61M)

### 4.3 Phân Tích Doanh Thu & Số Lượng
- **Theo thời gian:** doanh thu năm 2023 ổn định ~34 triệu/tháng; tháng 8/2023 đạt đỉnh; tháng 2/2024 giảm sâu do hiệu ứng hậu mùa lễ
- **Theo sản phẩm:** Electronics và Grocery chiếm tỷ trọng lớn; Pepsi và Samsung là thương hiệu dẫn đầu

### 4.4 Phân Tích Hành Vi Mua Hàng
- `Credit Card` tạo doanh thu cao nhất (~122M); `PayPal` thấp hơn đáng kể (~82M)
- Hơn 129.000 đơn đã giao thành công; còn nhiều đơn ở Shipped / Processing / Pending
- Phân tích `Ratings` và `Feedback` (Excellent / Average / Bad)

### 4.5 Phân Tích Vận Chuyển
- `Same-Day` dẫn đầu doanh thu (>141 triệu), cho thấy tốc độ giao hàng là yếu tố quan trọng
- Phân tích vận chuyển theo khu vực để cải thiện logistics

---

## 📖 Kể Chuyện Dữ Liệu

### Nguyên nhân ảnh hưởng đến doanh thu

| # | Vấn đề |
|---|--------|
| 1 | Doanh thu tập trung vào Electronics và Grocery; các danh mục khác đóng góp thấp |
| 2 | Hiệu suất thương hiệu không đồng đều (Pepsi, Samsung vs. Mitsubishi, BlueStar) |
| 3 | Một số loại sản phẩm có sức tiêu thụ thấp (Snack, BlueStar AC...) |
| 4 | Phương thức giao hàng và thanh toán tạo rào cản hoặc thúc đẩy quyết định mua |
| 5 | Tỷ lệ đơn hàng chưa hoàn tất còn cao, ảnh hưởng trải nghiệm khách hàng |

### Hướng giải quyết
- **Cá nhân hóa:** xây dựng chương trình ưu đãi theo phân khúc tuổi, thu nhập
- **Tối ưu danh mục:** đẩy mạnh sản phẩm doanh thu cao, tái cơ cấu sản phẩm kém hiệu quả
- **Cải thiện vận hành:** nâng cao chất lượng giao hàng nhanh, đa dạng hóa thanh toán

---

## 🤖 Mô Hình Học Máy

### K-Means Phân Cụm Khách Hàng
Phân khách hàng thành **4 cụm** dựa trên thu nhập, tổng chi tiêu, số lần mua và đánh giá:

| Cụm | Đặc điểm | Chiến lược đề xuất |
|-----|----------|--------------------|
| Cụm 0 | Thu nhập TB, mua nhiều, chi tiêu cao (~2.305) | Ưu đãi VIP, chương trình giữ chân |
| Cụm 1 | Thu nhập TB, mua ít, chi tiêu thấp (~666) | Kích cầu bằng flash sale, khuyến mãi |
| Cụm 2 | Thu nhập cao, chi tiêu trung bình (~1.370) | Sản phẩm cao cấp, dịch vụ cá nhân hóa |
| Cụm 3 | Thu nhập thấp, chi tiêu trung bình (~1.370) | Chương trình khách hàng thân thiết |

### Prophet Dự Báo Doanh Thu 6 Tháng (03/2024 – 09/2024)
- Xu hướng doanh thu **giảm nhẹ đều đặn** trong giai đoạn dự báo
- **MAPE < 3%** → độ chính xác tốt trong ngắn hạn (30–60 ngày)
- Cảnh báo: cần các chương trình kích cầu để ngăn đà giảm

---

## ✅ Kết Luận

Dự án hoàn thành đầy đủ quy trình: tiền xử lý → EDA → data storytelling → mô hình hóa. Các kết quả chính:
- Xác định rõ các phân khúc khách hàng tiềm năng qua K-Means
- Cảnh báo sớm xu hướng giảm doanh thu qua mô hình Prophet
- Đề xuất chiến lược: cá nhân hóa ưu đãi, tối ưu danh mục sản phẩm, cải thiện logistics và chăm sóc khách hàng trung thành

---

*TP. Hồ Chí Minh, 2025*

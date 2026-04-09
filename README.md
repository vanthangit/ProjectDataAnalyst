# \# 📊 Câu Chuyện Dữ Liệu Về Ngành Bán Lẻ

# \## 📌 Giới Thiệu

# 

# Dự án phân tích dữ liệu kinh doanh từ tập dữ liệu \*\*Superstore Sales\*\* – mô phỏng hoạt động bán lẻ của một công ty tại Hoa Kỳ. Thông qua các kỹ thuật \*\*trực quan hóa dữ liệu\*\*, dự án làm rõ mối quan hệ giữa doanh thu, lợi nhuận, chiết khấu, khu vực địa lý và hành vi khách hàng.

# 

# \---

# 

# \## ❓ Câu Hỏi Nghiên Cứu

# 

# \- Khu vực địa lý có ảnh hưởng đến doanh thu và lợi nhuận không?

# \- Các phân khúc khách hàng (Consumer, Corporate, Home Office) có xu hướng chi tiêu như thế nào?

# \- Có tồn tại nhóm sản phẩm "bán chạy nhưng không có lãi" không?

# \- Mức chiết khấu có tương quan với lợi nhuận như thế nào?

# \- Có thể phân loại khách hàng/sản phẩm để tối ưu chiến lược không?

# 

# \---

# 

# \## 📂 Dữ Liệu

# 

# \*\*Nguồn:\*\* \[Superstore Sales – Kaggle](https://www.kaggle.com/datasets/ishanshrivastava28/superstore-sales)

# 

# \*\*Các trường chính:\*\*

# 

# | Cột | Mô tả |

# |-----|-------|

# | `Order ID` | Mã đơn hàng |

# | `Order Date` / `Ship Date` | Ngày đặt / giao hàng |

# | `Ship Mode` | Phương thức vận chuyển |

# | `Segment` | Phân khúc khách hàng |

# | `Region` / `State` | Khu vực / Bang |

# | `Category` / `Sub-Category` | Danh mục / Danh mục con sản phẩm |

# | `Sales` | Doanh thu |

# | `Quantity` | Số lượng |

# | `Discount` | Mức chiết khấu |

# | `Profit` | Lợi nhuận |

# 

# \---

# 

# \## 🔧 Tiền Xử Lý Dữ Liệu

# 

# \- Xử lý giá trị null và dữ liệu trùng lặp

# \- Chuyển đổi `Order Date`, `Ship Date` sang kiểu `DateTime`

# \- Thêm cột `Year`, `Month`, `Day` phục vụ trực quan hóa

# \- Xóa các cột không cần thiết

# \- Giữ nguyên outliers vì phản ánh các trường hợp thực tế có giá trị

# 

# \---

# 

# \## 📖 Nội Dung Phân Tích

# 

# \### 1. Đặt Vấn Đề

# Doanh thu tăng trưởng nhưng lợi nhuận không tương xứng – dấu hiệu của hàng loạt sai lệch chiến lược.

# 

# \### 2. Nguyên Nhân Được Xác Định

# 

# | # | Vấn đề |

# |---|--------|

# | 1 | Quá nhiều đơn hàng lỗ |

# | 2 | Lợi nhuận không tương xứng với doanh thu |

# | 3 | Giảm giá sâu không hiệu quả |

# | 4 | Các Sub-Category kém hiệu quả (Tables, Bookcases, Supplies) |

# | 5 | Chi phí vận chuyển quá cao |

# | 6 | Chiến lược khách hàng chưa rõ ràng |

# | 7 | Tổn thất tập trung ở một số bang (Texas, Ohio, Pennsylvania) |

# 

# \### 3. Phân Tích Phân Khúc Khách Hàng

# 

# \- \*\*Consumer:\*\* Chiếm >50% đơn hàng nhưng tỷ suất lợi nhuận thấp nhất (11,5%)

# \- \*\*Corporate:\*\* Biên lợi nhuận cao hơn (13%), tiềm năng chưa được khai thác

# \- \*\*Home Office:\*\* Tỷ suất cao nhất (14%) nhưng thiếu quy mô

# 

# \### 4. Hướng Giải Quyết

# 

# 1\. Thiết lập hệ thống kiểm soát đơn hàng lỗ theo thời gian thực

# 2\. Phân tầng chiết khấu theo nhóm sản phẩm và khả năng sinh lời

# 3\. Tái cấu trúc danh mục sản phẩm theo hiệu quả tài chính

# 4\. Tùy biến phương thức vận chuyển theo đặc điểm đơn hàng

# 5\. Xây dựng chiến lược khách hàng phân tầng rõ ràng

# 6\. Điều chỉnh hoạt động riêng theo từng bang/khu vực địa lý

# 

# \---

# 

# \## 🏁 Kết Luận

# 

# Superstore đang rơi vào tình trạng \*\*bán hàng tăng nhưng lợi nhuận giảm\*\*. Giải pháp không phải là cắt giảm chi phí hay bán nhiều hơn, mà là một \*\*cuộc tái cấu trúc chiến lược toàn diện\*\*: thông minh hơn trong giảm giá, cá nhân hóa theo vùng, chọn đúng sản phẩm và chăm sóc đúng khách hàng.

# 


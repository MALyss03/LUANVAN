
# RULE DICTIONARY - QUY TẮC HỌC VỤ & PHÂN TÍCH RỦI RO

### 1. OFFICIAL CTU RULES (Quy chế Chính thức QĐ 3266/QĐ-ĐHCT)

* **`R_OFFICIAL_01`**: $ĐTBCHK < 0.80$ ở học kỳ đầu tiên $\rightarrow$ Cảnh báo học vụ.
* **`R_OFFICIAL_02`**: $ĐTBCHK < 1.00$ từ học kỳ 2 trở đi $\rightarrow$ Cảnh báo học vụ.
* **`R_OFFICIAL_03`**: Số tín chỉ rớt học kỳ $> 50\%$ tổng số tín chỉ đăng ký $\rightarrow$ Cảnh báo học vụ.
* **`R_OFFICIAL_04`**: Tổng số tín chỉ nợ đọng chưa tích lũy $> 24$ tín chỉ $\rightarrow$ Cảnh báo học vụ.

### 2. RESEARCH DERIVED INDICATORS (Chỉ số Nghiên cứu Dự báo)

* **`R_RESEARCH_01`**: $GPA\_Decline\_Streak \ge 3$ (Suy giảm GPA học kỳ 3 kỳ liên tiếp).
* **`R_RESEARCH_04`**: $TC\_Registration\_Decline\_Streak \ge 2$ (Giảm khối lượng đăng ký học 2 kỳ liên tiếp - Registration Load Declining).

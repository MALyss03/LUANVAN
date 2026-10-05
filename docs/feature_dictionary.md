
# FEATURE DICTIONARY - STUDENTPROGRESSAI (SEMESTER STUDENT PROFILE)

| Feature Name                   | Ý nghĩa                           | Công thức / Cấu trúc                              | Đơn vị   | Source     | ML Safe? |
| :----------------------------- | :---------------------------------- | :---------------------------------------------------- | :---------- | :--------- | :------- |
| `GPA_HK_4`                   | GPA Học kỳ thang 4                | $\sum (Điểm_4 \times TC) / \sum TC_{HK}$          | Thang 4     | Calculated | Yes      |
| `GPA_TL_4`                   | GPA Tích lũy thang 4              | $\sum (Điểm_4 \times TC_{Đạt}) / \sum TC_{TL}$  | Thang 4     | Calculated | Yes      |
| `Delta_GPA_HK`               | Mức biến động GPA HK            | $GPA\_HK_t - GPA\_HK_{t-1}$                         | Thang 4     | Temporal   | Yes      |
| `GPA_HK_Slope`               | Xu hướng ĐTBCHK                  | Linear Regression Slope trên lịch sử GPA HK        | Hệ số     | Temporal   | Yes      |
| `GPA_Decline_Streak`         | Chuỗi giảm GPA liên tục         | Số học kỳ liên tiếp$GPA\_HK_t < GPA\_HK_{t-1}$ | Học kỳ    | Temporal   | Yes      |
| `Credit_Deficit`             | Thâm hụt tín chỉ chuẩn         | $TC_{Chuẩn} - TC_{Tích Lũy}$                     | Tín chỉ   | Calculated | Yes      |
| `Prerequisite_Block_Count`   | Số môn bắt buộc bị khóa       | Đếm số môn chưa đỗ tiên quyết                | Số lượng | Curriculum | Yes      |
| `Graduation_Feasibility_Gap` | Khoảng cách khả thi tốt nghiệp | $Required\_Avg\_Load - Historical\_Avg\_Load$       | TC/HK       | Computed   | Yes      |

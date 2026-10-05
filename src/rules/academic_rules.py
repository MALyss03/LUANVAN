"""
===============================================================================
FILE: src/rules/academic_rules.py
MỤC ĐÍCH: Khởi tạo Bộ Quy tắc (Rule Engine) Phân loại Trạng thái Học tập
          & Đánh giá Rủi ro Học vụ theo Quy chế CTU và Chỉ số Nghiên cứu
===============================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Rule:
    rule_id: str
    rule_name: str
    feature: str
    operator: str
    threshold: Any
    source: str       # OFFICIAL_CTU_RULE | RESEARCH_DERIVED | HEURISTIC
    severity: str     # HIGH | MEDIUM | LOW
    description: str

    def evaluate(self, value: Any) -> bool:
        if value is None:
            return False
        if self.operator == ">=":
            return value >= self.threshold
        elif self.operator == ">":
            return value > self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "==":
            return value == self.threshold
        elif self.operator == "!=":
            return value != self.threshold
        return False


class RuleEngine:
    def __init__(self):
        self.rules: List[Rule] = [
            # Quy chế Chính thức CTU (QĐ 3266/QĐ-ĐHCT)
            Rule("R_OFFICIAL_01", "ĐTBCHK dưới ngưỡng cảnh báo học kỳ 1", "GPA_HK_4", "<", 0.80, "OFFICIAL_CTU_RULE", "HIGH", "ĐTBCHK < 0.80 ở học kỳ đầu tiên"),
            Rule("R_OFFICIAL_02", "ĐTBCHK dưới ngưỡng cảnh báo từ học kỳ 2", "GPA_HK_4", "<", 1.00, "OFFICIAL_CTU_RULE", "HIGH", "ĐTBCHK < 1.00 từ học kỳ 2 trở đi"),
            Rule("R_OFFICIAL_03", "Tỷ lệ tín chỉ không đạt HK vượt 50%", "Fail_Rate_HK", ">", 0.50, "OFFICIAL_CTU_RULE", "HIGH", "Số tín chỉ không đạt trong học kỳ vượt quá 50% số TC đăng ký"),
            Rule("R_OFFICIAL_04", "Tổng tín chỉ nợ đọng quá 24 TC", "TC_No_Tich_Luy", ">", 24, "OFFICIAL_CTU_RULE", "HIGH", "Tổng số tín chỉ nợ chưa tích lũy vượt quá 24 tín chỉ"),
            Rule("R_OFFICIAL_05", "ĐTBCTL dưới ngưỡng năm học", "GPA_Safety_Margin", "<", 0.0, "OFFICIAL_CTU_RULE", "HIGH", "ĐTBCTL không đạt mức tối thiểu theo trình độ năm học"),

            # Chỉ số Nghiên cứu / Phát hiện Sớm Rủi ro (Research Derived)
            Rule("R_RESEARCH_01", "GPA học kỳ giảm liên tục 3 kỳ", "GPA_Decline_Streak", ">=", 3, "RESEARCH_DERIVED", "MEDIUM", "ĐTBC học kỳ suy giảm liên tiếp trong 3 học kỳ"),
            Rule("R_RESEARCH_02", "Xu hướng GPA học kỳ âm sâu", "GPA_HK_Slope", "<=", -0.30, "RESEARCH_DERIVED", "MEDIUM", "Tốc độ sụt giảm điểm học kỳ đạt mức cảnh báo"),
            Rule("R_RESEARCH_03", "Tỷ lệ điểm thấp (D/D+/F) cao", "Low_Grade_Rate_HK", ">=", 0.35, "RESEARCH_DERIVED", "MEDIUM", "Chiếm hơn 35% tổng số học phần trong học kỳ"),
            Rule("R_RESEARCH_04", "Xu hướng giảm tải khối lượng đăng ký", "TC_Registration_Decline_Streak", ">=", 2, "RESEARCH_DERIVED", "LOW", "Số tín chỉ đăng ký giảm liên tục 2 học kỳ (Registration Load Declining)"),
            Rule("R_RESEARCH_05", "Khối lượng tín chỉ bắt buộc bị khóa cao", "Prerequisite_Block_Count", ">=", 3, "RESEARCH_DERIVED", "MEDIUM", "Có từ 3 học phần bắt buộc trở lên bị khóa do chưa đạt môn tiên quyết"),

            # Chỉ số Heuristic Tốt nghiệp
            Rule("R_HEURISTIC_01", "Tốc độ học tập không đủ tốt nghiệp đúng hạn", "Graduation_Feasibility_Gap", ">", 5.0, "HEURISTIC", "HIGH", "Khối lượng TC cần học mỗi kỳ vượt quá khả năng đăng ký lịch sử 5 TC/HK")
        ]

    def evaluate_student_semester(self, semester_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidences = []
        for rule in self.rules:
            val = semester_data.get(rule.feature)
            if rule.evaluate(val):
                evidences.append({
                    "rule_id": rule.rule_id,
                    "rule_name": rule.rule_name,
                    "feature": rule.feature,
                    "value": val,
                    "threshold": rule.threshold,
                    "source": rule.source,
                    "severity": rule.severity,
                    "description": rule.description
                })
        return evidences
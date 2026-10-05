"""
===============================================================================
FILE: src/data_engineering/curriculum_engine.py
MỤC ĐÍCH: Quản lý Mã môn học, Cây môn tiên quyết (Prerequisite Graph),
          Tiến độ Môn Tự chọn & Phân tích Trạng thái Đăng ký
===============================================================================
"""

import os
import pandas as pd
from typing import Dict, List, Set, Tuple

class CurriculumEngine:
    def __init__(self, prereq_csv_path: str = None, catalog_csv_path: str = None):
        self.prereq_map: Dict[str, List[str]] = {}
        self.course_catalog: Dict[str, Dict] = {}
        
        if prereq_csv_path and os.path.exists(prereq_csv_path):
            df_p = pd.read_csv(prereq_csv_path)
            for _, row in df_p.iterrows():
                code = str(row['course_code']).strip().upper()
                pre = str(row['prerequisite_code']).strip().upper() if pd.notna(row['prerequisite_code']) else ""
                if pre:
                    self.prereq_map.setdefault(code, []).append(pre)
        
        if catalog_csv_path and os.path.exists(catalog_csv_path):
            df_c = pd.read_csv(catalog_csv_path)
            for _, row in df_c.iterrows():
                code = str(row['course_code']).strip().upper()
                self.course_catalog[code] = {
                    'course_name': row.get('course_name', code),
                    'credits': int(row.get('credits', 3)),
                    'course_type': row.get('course_type', 'MANDATORY'),
                    'elective_group': row.get('elective_group', None),
                    'recommended_semester': int(row.get('recommended_semester', 1))
                }

    def get_course_status(self, course_code: str, passed_courses: Set[str], failed_courses: Set[str]) -> str:
        """Xác định 1 trong 6 trạng thái môn học chính xác."""
        course_code = course_code.upper()
        if course_code in passed_courses:
            return "PASSED"
        
        # Kiểm tra môn tiên quyết
        prereqs = self.prereq_map.get(course_code, [])
        for p in prereqs:
            if p not in passed_courses:
                return "BLOCKED_BY_PREREQUISITE"
                
        if course_code in failed_courses:
            return "FAILED"
            
        info = self.course_catalog.get(course_code, {})
        ctype = info.get('course_type', 'MANDATORY')
        if ctype == 'ELECTIVE':
            return "ELECTIVE_NOT_SELECTED"
        elif ctype == 'CONDITION':
            return "NOT_REQUIRED"
            
        return "NOT_TAKEN"

    def calculate_prerequisite_blocks(self, passed_courses: Set[str]) -> Tuple[int, int, int]:
        """Tính số môn bắt buộc bị khóa, tổng số TC bị khóa và độ dài chuỗi khóa dài nhất."""
        blocked_count = 0
        blocked_credits = 0
        max_chain_len = 0

        for code, info in self.course_catalog.items():
            if info.get('course_type') == 'MANDATORY' and code not in passed_courses:
                prereqs = self.prereq_map.get(code, [])
                if any(p not in passed_courses for p in prereqs):
                    blocked_count += 1
                    blocked_credits += info.get('credits', 3)
                    
                    # Tính độ dài chuỗi phụ thuộc bị khóa
                    chain_len = 1
                    curr = code
                    while curr in self.prereq_map:
                        next_nodes = [k for k, v in self.prereq_map.items() if curr in v]
                        if not next_nodes:
                            break
                        chain_len += 1
                        curr = next_nodes[0]
                    max_chain_len = max(max_chain_len, chain_len)

        return blocked_count, blocked_credits, max_chain_len
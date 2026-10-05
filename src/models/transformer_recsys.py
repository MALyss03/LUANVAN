"""
===============================================================================
FILE: src/models/transformer_recsys.py
MỤC ĐÍCH: Kiến trúc Student Transformer RecSys hỗ trợ Phân loại 4 Lớp Trạng thái
          & Tự động xử lý Lỗi Mất cân bằng Kích thước (Size Mismatch) khi Load Checkpoint
===============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 20):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class StudentTransformerRecSys(nn.Module):
    def __init__(
        self,
        feature_dim: int = 15,     # 15 đặc trưng mở rộng
        num_classes: int = 4,      # 4 Lớp: 0=On Track, 1=Delayed, 2=Warning, 3=Dismissal
        num_courses: int = 50,
        num_majors: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 20,          # Khớp với checkpoint [1, 20, 64]
        risk_hidden_dim: int = 32,  # Khớp với checkpoint [32, 80]
    ):
        super(StudentTransformerRecSys, self).__init__()

        self.major_embedding = nn.Embedding(num_majors, 16)
        self.semester_input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Head Dự báo Trạng thái Học vụ (Mặc định 4 lớp)
        self.risk_prediction_head = nn.Sequential(
            nn.Linear(d_model + 16, risk_hidden_dim),
            nn.ReLU(),
            nn.Linear(risk_hidden_dim, num_classes)
        )

        # Head Dự báo GPA Học kỳ tiếp theo
        self.gpa_prediction_head = nn.Sequential(
            nn.Linear(d_model + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Sequential Course RecSys Head
        self.course_embedding = nn.Embedding(num_courses, d_model)
        self.fusion_fc = nn.Sequential(
            nn.Linear((d_model + 16) + d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Ghi đè load_state_dict chuẩn của PyTorch để tự động ghép nối trọng số
        khi kích thước không khớp (Size Mismatch) do nâng từ 2 lớp lên 4 lớp.
        """
        # Bóc tách state_dict nếu được bọc trong dict con
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

        model_dict = super().state_dict()
        adapted_dict = {}

        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    adapted_dict[k] = v
                else:
                    print(
                        f"⚠️ Phát hiện lệch kích thước tại '{k}': Checkpoint {list(v.shape)} vs "
                        f"Mô hình hiện tại {list(model_dict[k].shape)}. Đang tự động ghép trọng số..."
                    )
                    # Tự động ghép trọng số Linear 2D (chuyển đổi từ 2 lớp sang 4 lớp)
                    if v.dim() == 2 and model_dict[k].dim() == 2:
                        min_out = min(v.shape[0], model_dict[k].shape[0])
                        min_in = min(v.shape[1], model_dict[k].shape[1])
                        temp_tensor = model_dict[k].clone()
                        temp_tensor[:min_out, :min_in] = v[:min_out, :min_in]
                        adapted_dict[k] = temp_tensor
                    # Tự động ghép Bias 1D
                    elif v.dim() == 1 and model_dict[k].dim() == 1:
                        min_out = min(v.shape[0], model_dict[k].shape[0])
                        temp_tensor = model_dict[k].clone()
                        temp_tensor[:min_out] = v[:min_out]
                        adapted_dict[k] = temp_tensor
            else:
                print(f"⚠️ Bỏ qua trọng số không khớp trong mô hình: {k}")

        model_dict.update(adapted_dict)
        return super().load_state_dict(model_dict, strict=False)

    def load_checkpoint(self, checkpoint_path_or_dict, device='cpu'):
        """
        Hàm nạp checkpoint tiện ích từ đường dẫn file hoặc dictionary.
        """
        if isinstance(checkpoint_path_or_dict, (str, os.PathLike)):
            if not os.path.exists(checkpoint_path_or_dict):
                print(f"⚠️ File checkpoint không tồn tại: {checkpoint_path_or_dict}")
                return False
            print(f"📥 Đang tải checkpoint từ: {checkpoint_path_or_dict}")
            state_dict = torch.load(checkpoint_path_or_dict, map_location=device)
        else:
            state_dict = checkpoint_path_or_dict

        self.load_state_dict(state_dict, strict=False)
        print("✅ Đã nạp trọng số mô hình thành công!")
        return True

    def forward(
        self,
        semester_seq: torch.Tensor,
        candidate_course_ids: torch.Tensor,
        major_ids: torch.Tensor = None,
    ) -> dict:
        batch_size = semester_seq.size(0)

        if major_ids is None:
            major_ids = torch.zeros(batch_size, dtype=torch.long, device=semester_seq.device)

        x_proj = self.semester_input_proj(semester_seq)
        x_pos = self.pos_encoder(x_proj)
        transformer_out = self.transformer_encoder(x_pos)

        # Representation vector của học kỳ gần nhất
        z_s_seq = transformer_out[:, -1, :]
        z_m = self.major_embedding(major_ids)
        z_s = torch.cat([z_s_seq, z_m], dim=-1)

        # Logits phân loại 4 lớp & Xác suất Softmax
        risk_logits = self.risk_prediction_head(z_s)
        risk_probs = F.softmax(risk_logits, dim=-1)
        
        pred_gpa = self.gpa_prediction_head(z_s)

        # Gợi ý môn học có hiệu chỉnh theo rủi ro học vụ
        z_c = self.course_embedding(candidate_course_ids)
        num_candidates = z_c.size(1)
        
        # Tối ưu hóa bộ nhớ
        z_s_expanded = z_s.unsqueeze(1).expand(-1, num_candidates, -1)

        fusion_input = torch.cat([z_s_expanded, z_c], dim=-1)
        raw_rec_scores = self.fusion_fc(fusion_input).squeeze(-1)

        # Phạt điểm môn học đối với sinh viên có nguy cơ rủi ro cao (Lớp 2 & 3)
        high_risk_penalty = (risk_probs[:, 2] + risk_probs[:, 3]).unsqueeze(1) * 2.0
        adjusted_rec_scores = raw_rec_scores - high_risk_penalty

        return {
            'risk_logits': risk_logits,
            'risk_probs': risk_probs,
            'pred_class': torch.argmax(risk_probs, dim=-1),
            'pred_gpa': pred_gpa.squeeze(-1),
            'rec_scores': adjusted_rec_scores,
            'z_s': z_s,
        }
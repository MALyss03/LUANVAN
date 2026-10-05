"""
===============================================================================
FILE: src/models/train_and_explain.py
MỤC ĐÍCH: Huấn luyện Mô hình Phân loại 4 Lớp & Phân tích Trọng số Đặc trưng
===============================================================================
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer_recsys import StudentTransformerRecSys


def generate_mock_15_features_dataset(num_students: int = 150, max_semesters: int = 6):
    """Giả lập dữ liệu chuỗi 15 đặc trưng cho 4 lớp phân loại."""
    np.random.seed(42)
    feature_dim = 15
    X_seq = np.random.randn(num_students, max_semesters, feature_dim).astype(np.float32)

    # Giả lập nhãn mục tiêu 4 lớp (0, 1, 2, 3)
    y_target = np.random.choice([0, 1, 2, 3], size=num_students, p=[0.5, 0.25, 0.15, 0.10])
    y_gpa = np.clip(np.random.normal(2.8, 0.6, size=num_students), 0.0, 4.0).astype(np.float32)
    major_ids = np.random.choice([0, 1, 2], size=num_students)

    return X_seq, major_ids, y_target, y_gpa


def run_train(checkpoint_path: str = None):
    # Tự động chọn thiết bị tính toán (GPU nếu có)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Đang sử dụng thiết bị: {device}")

    print("\n==================================================")
    print(" 1. KHỞI TẠO DỮ LIỆU & MÔ HÌNH TRANSFORMER 4 LỚP")
    print("==================================================")
    
    X_seq, major_ids, y_target, y_gpa = generate_mock_15_features_dataset()

    tensor_X = torch.tensor(X_seq, dtype=torch.float32).to(device)
    tensor_majors = torch.tensor(major_ids, dtype=torch.long).to(device)
    tensor_y_target = torch.tensor(y_target, dtype=torch.long).to(device)
    tensor_y_gpa = torch.tensor(y_gpa, dtype=torch.float32).to(device)

    # Khởi tạo mô hình 4 lớp
    model = StudentTransformerRecSys(
        feature_dim=15,
        num_classes=4,
        num_courses=50,
        num_majors=3,
        max_len=20,
        risk_hidden_dim=32
    ).to(device)

    # Nếu không chỉ định checkpoint, tự động kiểm tra ưu tiên checkpoint 4 lớp -> 2 lớp
    models_dir = PROJECT_ROOT / "models"
    if checkpoint_path is None:
        ckpt_4_classes = models_dir / "transformer_recsys_4classes.pth"
        ckpt_2_classes = models_dir / "transformer_recsys.pth"
        
        if ckpt_4_classes.exists():
            checkpoint_path = str(ckpt_4_classes)
        elif ckpt_2_classes.exists():
            checkpoint_path = str(ckpt_2_classes)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n-> Tiến hành nạp weights từ checkpoint: {checkpoint_path}")
        model.load_checkpoint(checkpoint_path, device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    criterion_class = nn.CrossEntropyLoss()
    criterion_gpa = nn.MSELoss()

    print("\n==================================================")
    print(" 2. BẮT ĐẦU HUẤN LUYỆN (TRAINING LOOP)")
    print("==================================================")
    model.train()
    epochs = 15
    dummy_courses = torch.randint(0, 30, (len(tensor_X), 5), device=device)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        outputs = model(tensor_X, dummy_courses, tensor_majors)
        
        loss_class = criterion_class(outputs['risk_logits'], tensor_y_target)
        loss_gpa = criterion_gpa(outputs['pred_gpa'], tensor_y_gpa)

        total_loss = loss_class + 0.5 * loss_gpa
        total_loss.backward()
        optimizer.step()

        if epoch % 3 == 0 or epoch == 1:
            acc = (outputs['pred_class'] == tensor_y_target).float().mean().item() * 100
            print(
                f"Epoch [{epoch:02d}/{epochs}] - Total Loss: {total_loss.item():.4f} | "
                f"Class Loss: {loss_class.item():.4f} | Accuracy: {acc:.1f}%"
            )

    # Lưu mô hình 4 lớp chuẩn sau khi train
    os.makedirs(models_dir, exist_ok=True)
    save_path = models_dir / "transformer_recsys_4classes.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n-> Đã lưu mô hình tại: {save_path}")

    print("\n==================================================")
    print(" 3. ĐÁNH GIÁ MỨC ĐỘ ẢNH HƯỞNG CỦA CÁC ĐẶC TRƯNG")
    print("==================================================")
    feature_names = [
        'Current_Semester', 'GPA_TL', 'GPA_HK', 'GPA_Trend',
        'Credits_Earned', 'Credit_Deficit', 'Credits_Failed', 'Fail_Rate_HK',
        'Total_Repeat_Count', 'Prerequisite_Deficit_Count', 'GDTC_Completed',
        'GDQP_Completed', 'English_Completed', 'Consecutive_Warnings', 'GPA_Safety_Margin'
    ]

    model.eval()
    with torch.no_grad():
        weights = model.semester_input_proj.weight.detach().cpu().abs().mean(dim=0).numpy()
        weights_normalized = weights / weights.sum()

    for fname, w in zip(feature_names, weights_normalized):
        print(f"  - {fname:<28}: {w*100:5.1f}% đóng góp")


if __name__ == '__main__':
    ckpt_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_train(checkpoint_path=ckpt_file)
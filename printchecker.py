import os
import requests
import base64
import json
import re
from pathlib import Path
from PIL import Image, ImageEnhance
import pandas as pd

# Clova OCR 설정
secret_key = ""
api_url = ""

# 학생 목록 로드 (학번, 이름 기준 매칭용)
student_df = pd.read_csv("student_list.csv")
student_dict = {str(row['학번']).strip(): row['이름'].strip() for _, row in student_df.iterrows()}
name_dict = {row['이름'].strip(): str(row['학번']).strip() for _, row in student_df.iterrows()}

# 제외할 단어 리스트
exclude_names = {"학번", "성명", "제출자", "이름", "다음", "프린트", "문제"}

def crop_image(image_path, output_path, height=350):
    with Image.open(image_path) as img:
        cropped = img.crop((0, 0, img.width, height))

        # 대비 + 밝기 동시에 조정
        contrast_enhanced = ImageEnhance.Contrast(cropped).enhance(2.0)
        brightened = ImageEnhance.Brightness(contrast_enhanced).enhance(1.5)

        brightened.save(output_path)

def ocr_image(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    headers = {
        "X-OCR-SECRET": secret_key,
        "Content-Type": "application/json"
    }

    data = {
        "version": "V2",
        "requestId": "sample_id",
        "timestamp": 0,
        "images": [{"format": "jpg", "data": image_data, "name": "sample"}]
    }

    response = requests.post(api_url, headers=headers, json=data)
    result = response.json()
    fields = []
    for field in result.get("images", [])[0].get("fields", []):
        fields.append(field["inferText"])
    return fields

def extract_info(texts):
    joined = " ".join(texts)
    print_type = None
    student_id = None
    name = None

    # 프린트 종류 예: pp.011~018
    match_print = re.search(r"pp\.?\d{3}~\d{3}", joined, re.IGNORECASE)
    if match_print:
        print_type = match_print.group().replace(",", "").strip()

    # 학번 탐색
    id_match = re.search(r"학번\D*(\d{5,6})", joined)
    if id_match:
        student_id = id_match.group(1)
    else:
        fallback_id = re.search(r"\b\d{5,6}\b", joined)
        if fallback_id:
            student_id = fallback_id.group()

    # 이름 탐색
    name_match = re.search(r"(이름|성명)\s*[:\-]?\s*([가-힣]{2,4})", joined)
    if name_match:
        name = name_match.group(2)
    else:
        name_candidates = re.findall(r"[가-힣]{2,4}", joined)
        for candidate in name_candidates:
            if candidate not in exclude_names:
                name = candidate
                break

    # 유효성 체크 및 보완
    if name not in name_dict:
        name = ""
    if student_id not in student_dict:
        student_id = ""

    # 상호 보완
    if name in name_dict:
        if not student_id:
            student_id = name_dict[name]
        elif name_dict[name] != student_id:
            name = student_dict.get(student_id, "")
    elif student_id in student_dict:
        name = student_dict[student_id]
    else:
        name = ""
        student_id = ""

    return print_type, student_id, name

# 경로 설정
img_dir = Path("./image")
cropped_dir = Path("./cropped")
cropped_dir.mkdir(exist_ok=True)

results = []
failed_images = []

# 이미지 순회
for img_path in img_dir.glob("*.jpg"):
    cropped_path = cropped_dir / f"cropped_{img_path.name}"
    crop_image(img_path, cropped_path)
    try:
        texts = ocr_image(cropped_path)
        print_type, student_id, name = extract_info(texts)
        results.append({
            "파일명": img_path.name,
            "프린트 종류": print_type or "",
            "학번": student_id or "",
            "이름": name or ""
        })
        if not print_type or (not student_id and not name):
            failed_images.append({"파일명": img_path.name})
    except Exception:
        results.append({
            "파일명": img_path.name,
            "프린트 종류": "오류",
            "학번": "오류",
            "이름": "오류"
        })
        failed_images.append({"파일명": img_path.name})

# 결과 저장
result_df = pd.DataFrame(results)
result_df.to_csv("result.csv", index=False, encoding="utf-8-sig")

# 인식 실패한 이미지 저장
if failed_images:
    pd.DataFrame(failed_images).to_csv("failed_images.csv", index=False, encoding="utf-8-sig")
    print("⚠️ 인식 실패 파일 저장 완료: failed_images.csv")

# -------------------------------
# 학생별 프린트 종류 Pivot Table 생성
# -------------------------------
pivot_df = result_df[result_df["프린트 종류"].notna() & result_df["학번"].notna() & (result_df["프린트 종류"] != "") & (result_df["학번"] != "") & (result_df["이름"] != "")]

# 중복 제거
pivot_df = pivot_df.drop_duplicates(subset=["학번", "프린트 종류"])

# 이름과 학번 함께 보기 위해 하나의 식별자 열 추가
pivot_df["식별자"] = pivot_df["이름"] + " (" + pivot_df["학번"] + ")"

# 피벗 테이블 생성
submission_matrix = pivot_df.pivot_table(index="식별자", columns="프린트 종류", values="파일명", aggfunc="count", fill_value=0)
submission_matrix = submission_matrix.applymap(lambda x: "O" if x > 0 else "X")

# 열 정렬 (프린트 번호 기준 숫자 정렬)
sorted_columns = sorted(submission_matrix.columns, key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else x)
submission_matrix = submission_matrix[sorted_columns]

# 정렬 (이름, 학번 순)
submission_matrix = submission_matrix.reset_index().sort_values(by="식별자")

# 저장
submission_matrix.to_csv("student_vs_print_type.csv", index=False, encoding="utf-8-sig")
print("✅ 학생-프린트 제출 여부 매트릭스 저장 완료: student_vs_print_type.csv")

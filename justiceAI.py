import json
import numpy as np
import torch
import re
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import statistics

# =============================================
# 모델 및 디바이스 설정 (KLUE/BERT만 사용)
# =============================================
model_name = "klue/roberta-small"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# =============================================
# 데이터 로드: updated_laws.json (판결 데이터)
# =============================================
with open('/home/yechan/dataset/updated_laws.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 각 판결의 텍스트, 인용된 법률(true_laws), 사건 번호 추출
case_texts    = [entry["case_text"] for entry in data]
true_laws_all = [entry["true_laws"] for entry in data]
case_numbers  = [entry["case_number"] for entry in data]

# =============================================
# 법률 정보를 결합하여 판결문 재구성 함수
# (판결문 본문과 해당 판결에서 인용한 법률 정보를 결합)
# =============================================
def construct_judgment_text(case_text, true_laws):
    law_texts = []
    for law in true_laws:
        if isinstance(law, dict):
            name = law.get("law_name", "").strip()
            article = law.get("article_num", "").strip()
            content = law.get("content")
            if content is not None and str(content).strip():
                law_text = f"{name} {article} {str(content).strip()}"
            else:
                law_text = f"{name} {article}"
        else:
            law_text = str(law).strip()
        if law_text:
            law_texts.append(law_text)
    combined = case_text + "\n[인용 법률]\n" + "\n".join(law_texts) if law_texts else case_text
    return combined

# 각 판결마다, 판결문과 인용 법률 정보를 결합하여 재구성된 텍스트 구성
judgment_texts = [construct_judgment_text(case_texts[i], true_laws_all[i])
                  for i in range(len(case_texts))]

# =============================================
# 임베딩 함수: KLUE/BERT를 이용한 판결문 임베딩 (배치 처리, tqdm 사용)
# =============================================
def encode_texts(texts, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Judgments", ncols=80):
        batch = [text.strip() for text in texts[i:i+batch_size] if text.strip()]
        if not batch:
            continue
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            batch_emb = model(**inputs).pooler_output.cpu().numpy()
        embeddings.append(batch_emb)
    return np.vstack(embeddings) if embeddings else np.array([])

# =============================================
# 판결문 임베딩 계산 (한 번만 수행)
# =============================================
judgment_embeddings = encode_texts(judgment_texts, batch_size=128)

# =============================================
# 추천 함수: 입력 판결(legal_text) 임베딩과 전체 판결 임베딩 간의 코사인 유사도를 계산하여 추천
# =============================================
def recommend_judgments(query_text, top_k=5):
    query_embedding = encode_texts([query_text], batch_size=1)
    similarities = cosine_similarity(query_embedding, judgment_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            "case_number": case_numbers[idx],
            "input_legal_text": query_text,
            "recommended_case_number": case_numbers[idx],
            "recommended_case_text": case_texts[idx],
            "constructed_judgment_text": judgment_texts[idx],
            "similarity": float(similarities[idx]),
            "true_laws": true_laws_all[idx]
        })
    return recommendations

# =============================================
# 평가 함수: 추천된 판결들의 인용 법률과 입력 판결의 인용 법률 비교
# 평가 시, 입력 판결의 인용 법률 개수에 맞춰 추천 결과를 필터링하여 평가
# =============================================
def evaluate_recommendations(input_true_laws, recommended):
    # input_true_laws: 리스트 (각 dict에 "law_name")
    # recommended: 추천된 판결들의 상세 정보 리스트 (각 항목에 "true_laws")
    # 추출: 입력 판결의 true_laws에서 고유 법령명 (소문자)
    input_law_set = {law["law_name"].lower() for law in input_true_laws if law.get("law_name")}
    n_true = len(input_law_set)
    
    # 추천된 판결은 여러 건이 있을 수 있지만, 여기서는 추천된 판결들의 true_laws를
    # 모두 모아서 추천된 법령 집합으로 만듭니다.
    rec_law_set = set()
    # 또한, 추천 결과에 "similarity" 점수를 기준으로 정렬
    recommended_sorted = sorted(recommended, key=lambda x: x.get("similarity", 0), reverse=True)
    
    # 만약 입력 true_laws가 0개라면 평가할 수 없으므로, 기본값 0 반환
    if n_true == 0:
        return {"Precision": 0, "Recall": 0, "F1 Score": 0, "MRR": 0, "Recall@K": 0}
    
    # 추천 결과에서 상위 n_true 판결의 인용 법령들을 모으기
    filtered_rec_set = set()
    for rec in recommended_sorted[:n_true]:
        for law in rec.get("true_laws", []):
            if law.get("law_name"):
                filtered_rec_set.add(law["law_name"].lower())
    
    # 계산: precision, recall, f1
    matched = input_law_set.intersection(filtered_rec_set)
    matched_count = len(matched)
    precision = matched_count / len(filtered_rec_set) if filtered_rec_set else 0
    recall = matched_count / len(input_law_set) if input_law_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # MRR: 추천된 판결들 중, 첫 번째로 입력 법령과 일치하는 판결의 역순위
    mrr = 0
    rec_case_numbers = [rec["recommended_case_number"] for rec in recommended_sorted[:n_true]]
    for idx, case_no in enumerate(rec_case_numbers, start=1):
        # 각 추천 판결의 true_laws 집합
        rec_laws = {law["law_name"].lower() for law in rec.get("true_laws", []) if law.get("law_name")}
        if rec_laws.intersection(input_law_set):
            mrr = 1/idx
            break
    recall_at_k = recall  # 단순히 recall로 산정

    return {
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MRR": mrr,
        "Recall@K": recall_at_k
    }

# =============================================
# 테스트 실행: 입력 판결(legal_text)을 사용하여 관련 판결 추천 및 평가 (진행 상황 표시)
# =============================================
num_tests = 6000
results = []
all_precisions = []
all_recalls = []
all_f1s = []
all_mrrs = []
all_recall_at_ks = []

for _ in tqdm(range(num_tests), desc="Running Judgment Tests", ncols=80):
    # 입력 판결로 사용할 판결(legal_text) 선택
    random_idx = random.randint(0, len(judgment_texts) - 1)
    input_text = judgment_texts[random_idx]
    input_true_laws = true_laws_all[random_idx]
    input_case_number = case_numbers[random_idx]
    
    recommended = recommend_judgments(input_text, top_k=5)
    eval_metrics = evaluate_recommendations(input_true_laws, recommended)
    
    all_precisions.append(eval_metrics["Precision"])
    all_recalls.append(eval_metrics["Recall"])
    all_f1s.append(eval_metrics["F1 Score"])
    all_mrrs.append(eval_metrics["MRR"])
    all_recall_at_ks.append(eval_metrics["Recall@K"])
    
    results.append({
        "input_case_number": input_case_number,
        "input_legal_text": input_text,
        "input_true_laws": [law["law_name"] for law in input_true_laws if law.get("law_name")],
        "recommended": recommended,
        "evaluation": eval_metrics
    })

avg_metrics = {
    "Precision": statistics.mean(all_precisions),
    "Recall": statistics.mean(all_recalls),
    "F1 Score": statistics.mean(all_f1s),
    "MRR": statistics.mean(all_mrrs),
    "Recall@K": statistics.mean(all_recall_at_ks)
}

print("평균 평가 지표:")
print(avg_metrics)

# =============================================
# 결과 저장
# =============================================
output_file = "/home/yechan/dataset/judgment_recommendation_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({
        "results": results,
        "average_metrics": avg_metrics
    }, f, ensure_ascii=False, indent=4)

print("✅ 판결 추천 결과 저장 완료!")

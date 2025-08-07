🧾 Justice AI: Enhancing Legal Statute Recommendation through Structured Case References
📌 Overview
Justice AI is a legal AI system that recommends relevant legal statutes by analyzing Korean court decisions. It combines dense passage retrieval using KLUE-RoBERTa with structured legal reference evaluation to provide accurate legal recommendations.

🧠 Key Features
⚖️ Legal Statute Recommendation based on referenced cases

📚 Uses KLUE-RoBERTa for embedding legal text

🔍 Cosine similarity search to retrieve similar rulings

📊 Evaluation metrics: Precision, Recall, F1 Score, MRR, Recall@K

🧾 Handles Korean legal documents with structured case-law citations

🗂️ Saves results as structured JSON for further analysis

🗂️ Dataset Format (Input)
Each court decision in updated_laws.json should follow this structure:

json
복사
편집
{
  "case_number": "2022다12345",
  "case_text": "이 사건은 ...",
  "true_laws": [
    {
      "law_name": "민법",
      "article_num": "제750조",
      "content": "고의 또는 과실로 타인에게 손해를 가한 자는 그 손해를 배상할 책임이 있다."
    }
  ]
}
🚀 How to Run
bash
복사
편집
# Install dependencies
pip install torch transformers scikit-learn tqdm

# Run the recommendation & evaluation script
python data_recom_22.py
✅ Results will be saved at:

swift
/home/yechan/dataset/judgment_recommendation_results.json
📈 Output Format (Results)
Each entry includes recommendation results and evaluation metrics:

json
{
  "input_case_number": "2022다12345",
  "recommended": [
    {
      "recommended_case_number": "2020다67890",
      "similarity": 0.8731,
      "true_laws": [...]
    }
  ],
  "evaluation": {
    "Precision": 0.75,
    "Recall": 1.0,
    "F1 Score": 0.857,
    "MRR": 1.0,
    "Recall@K": 1.0
  }
}
🧪 Evaluation Summary
Total Test Cases: 87k

🏛️ Model Architecture
Default Model: klue/roberta-small

Compatible Models: Any Hugging Face Transformer model for sentence embedding, such as:

klue/bert-base

monologg/kobert

beomi/KcELECTRA-base

snunlp/KR-BERT-char16424

etc.

Embedding Method: Uses pooler_output or [CLS] token embedding from the model

Similarity Measure: cosine_similarity between sentence embeddings

Model Switch: Simply change the model_name variable in the script:

model_name = "klue/bert-base"  # Replace with any Hugging Face model name

📎 Project Structure
bash
├── JusticeAI.py               # Main code: embedding, recommendation, evaluation
├── updated_laws.json              # Input data: legal case documents
└── README.md

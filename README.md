# 🧾 Justice AI: Enhancing Legal Statute Recommendation through Structured Case References

## 📌 Overview  
**Justice AI** is a legal AI system that recommends relevant legal statutes by analyzing Korean court decisions.  
It combines **dense passage retrieval** (KLUE-RoBERTa or any Transformer-based model) with **structured legal reference evaluation** to provide accurate legal recommendations.

---

## 🧠 Key Features  
- ⚖️ Legal statute recommendation based on referenced legal cases  
- 📚 Uses KLUE-RoBERTa or other transformer models for legal text embedding  
- 🔍 Cosine similarity search to retrieve similar rulings  
- 📊 Provides evaluation metrics: `Precision`, `Recall`, `F1 Score`, `MRR`, `Recall@K`  
- 🧾 Handles Korean legal documents with structured case-law citations  
- 💾 Saves results in structured JSON format for further analysis  

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install torch transformers scikit-learn tqdm

# 2. Run the recommendation & evaluation script
python data_recom_22.py

```

---

## 🗂️ Dataset Format (Input)

Each court decision in `updated_laws.json` should follow this structure:

```bash
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
```
---

##📈 Output Format (Results)
Each entry includes recommendation results and evaluation metrics:

```json
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
```
---

## 🧪 Evaluation Summary
Total Test Cases: 87,000
---
##🏛️ Model Architecture
Default Model: klue/roberta-small

Compatible Models (easily switchable via model_name):

klue/bert-base

monologg/kobert

beomi/KcELECTRA-base

snunlp/KR-BERT-char16424

(or any Hugging Face encoder-based model)

<pre><code class="language-python"> # In the code, replace this line: model_name = "klue/roberta-small" # with any compatible model: model_name = "klue/bert-base" </code></pre>
Embedding: pooler_output or [CLS] vector

Similarity Measure: Cosine similarity between embeddings
---
📁 Project Structure
```bash
├── justiceAI.py                     # Main code: embedding, recommendation, evaluation
├── updated_laws.json                    # Input data: legal case documents
├── judgment_recommendation_results.json # Output: recommendation + evaluation
└── README.md                            # Project description

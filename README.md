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

## 🗂️ Dataset Format (Input)

Each court decision in `updated_laws.json` should follow this structure:

```json
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

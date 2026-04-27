# ESG-Banking
ESG Banking Evaluation API  An intelligent ESG (Environmental, Social, Governance) evaluation system built using FastAPI + NLP (spaCy) to automate sustainable finance decision-making.

This API enables banks and financial institutions to:

📄 Analyze ESG reports
💰 Evaluate financial health
🚫 Apply exclusion criteria
🧠 Generate automated loan decisions
🚀 Key Features

📂 1. Document Management
Upload and manage organization documents:
ESG Reports
Financial Reports
Exclusion Reports
Local storage simulation (can be extended to cloud like GCS/S3)

🧠 2. ESG Metrics Extraction (NLP Powered)
Uses spaCy + Matcher to extract:
🌍 Carbon emission reduction (%)
🌫️ GHG emission reduction
👥 Diversity index
🏛️ Governance score
♻️ Landfill reduction
🩺 Workplace safety improvements

💰 3. Financial Analysis
Automatically extracts:
Revenue (supports M/B units)
Profit per share
Evaluates:
Loss-making companies
Zero revenue cases

🚫 4. Exclusion Criteria Engine
Flags companies involved in:
Coal / fossil fuel activities
Nuclear / arms trade
Child labor / unethical practices
Gambling / pornography
Illegal deforestation
Sanctioned entities

⚖️ 5. Intelligent Decision Engine
Combines:
ESG performance
Financial health
Exclusion rules

Outputs:
✅ Approval / Rejection

💼 Loan type recommendation:
Best-in-Class Investment
Transition Loan
Thematic Investment

🛠️ Tech Stack
Backend: FastAPI
NLP Engine: spaCy (en_core_web_sm)
Pattern Matching: spaCy Matcher
Storage: Local (extendable to cloud)
Validation: Pydantic

📌 API Endpoints
🔹 Upload Document
POST /api/documents/upload

Form Data:
organization_id
document_type → esg | financial | exclusion
file

🔹 List Documents
GET /api/documents?organization_id=123

🔹 Trigger Evaluation
POST /api/evaluation

Form Data:
organization_id
esg_doc_id (optional)
financial_doc_id (optional)
exclusion_doc_id (optional)

🔹 Get Evaluation Summary
GET /api/evaluation/{evaluation_id}

🔹 Get Evaluation Reasons
GET /api/evaluation/{evaluation_id}/reasons

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/your-username/esg-evaluation-api.git
cd esg-evaluation-api

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install spaCy Model
python -m spacy download en_core_web_sm

4️⃣ Run the Server
uvicorn app:app --reload

5️⃣ Open API Docs
http://localhost:8000/docs

🧪 How It Works
Upload 3 documents:
ESG Report
Financial Report
Exclusion Report
NLP extracts key metrics
System checks:
ESG thresholds
Financial health
Exclusion rules
Generates:
Approval decision
Loan recommendation
Detailed reasoning

📊 Example Output
{
  "approved": true,
  "loan_option": "Transition Loan",
  "reasons": [
    "Carbon emission reduction meets threshold",
    "Financials are healthy"
  ]
}

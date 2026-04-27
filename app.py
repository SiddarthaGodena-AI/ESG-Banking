import os
import uuid
import re
import logging
from typing import List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ESG_Evaluation")

# Create FastAPI app
app = FastAPI(title="ESG Banking Evaluation API")

# Allow CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for simulated document storage (replace with GCS integration in production)
STORAGE_DIR = "storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# In-memory "database" for documents and evaluations.
documents_db = {}     # key: doc_id, value: metadata dict
evaluations_db = {}   # key: evaluation_id, value: evaluation dict

# Allowed document types for upload
ALLOWED_DOC_TYPES = {"esg", "financial", "exclusion"}

####################################
# Helper Functions for File Storage
####################################
def save_file(file: UploadFile, organization_id: str, doc_type: str) -> str:
    try:
        file_extension = file.filename.split(".")[-1]
        doc_id = str(uuid.uuid4())
        filename = f"{doc_id}.{file_extension}"
        file_path = os.path.join(STORAGE_DIR, filename)
        content = file.file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        documents_db[doc_id] = {
            "organization_id": organization_id,
            "document_type": doc_type,
            "filename": filename,
            "file_path": file_path,
        }
        logger.info(f"File saved: {filename} for Org {organization_id} as {doc_type}")
        return doc_id
    except Exception as e:
        logger.error("Error saving file: %s", e)
        raise HTTPException(status_code=500, detail="File upload failed")

def get_documents_by_org(organization_id: str) -> List[dict]:
    return [
        {**meta, "doc_id": doc_id}
        for doc_id, meta in documents_db.items()
        if meta["organization_id"] == organization_id
    ]

def read_document(doc_id: str) -> str:
    meta = documents_db.get(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        with open(meta["file_path"], "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error("Error reading document %s: %s", doc_id, e)
        raise HTTPException(status_code=500, detail="Error reading document")

####################################
# Advanced ESG Metrics Extraction Using spaCy
####################################
import spacy
from spacy.matcher import Matcher

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Define ESG criteria thresholds (adjust as needed)
ESG_CRITERIA = {
    'carbon_emission_reduction': 20,  # Minimum percent reduction required for carbon emissions
    'diversity_index': 0.5,           # Minimum diversity index required
    'governance_score': 70,           # Minimum governance score required
}

def extract_esg_metrics(text: str) -> dict:
    """
    Extract key ESG and sustainability metrics from report text using spaCy's NLP and Matcher.
    Looks for:
      - Basic carbon emission reduction (e.g. "25% reduction in carbon emissions")
      - Absolute GHG emissions reduction (e.g. "35% reduction in absolute GHG emissions")
      - Diversity index (e.g. "diversity index of 0.55")
      - Governance score (e.g. "governance score of 75")
      - Landfill intensity reduction (e.g. "44% reduction in landfill intensity")
      - Recordable injury frequency improvement (e.g. "25% improvement in recordable injury frequency")
    """
    metrics = {
        "carbon_emission_reduction": 0,
        "ghg_emission_reduction": 0,
        "diversity_index": 0.0,
        "governance_score": 0,
        "landfill_intensity_reduction": 0,
        "recordable_injury_frequency_improvement": 0,
    }
    
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    pattern_carbon = [
        {"LIKE_NUM": True},
        {"TEXT": "%"},
        {"LOWER": "reduction"},
        {"LOWER": "in"},
        {"LOWER": "carbon"},
        {"LOWER": "emissions"}
    ]
    pattern_ghg = [
        {"LIKE_NUM": True},
        {"TEXT": "%"},
        {"LOWER": "reduction"},
        {"LOWER": "in"},
        {"LOWER": {"IN": ["absolute", "total"]}},
        {"LOWER": "ghg"},
        {"LOWER": {"IN": ["emissions", "greenhouse"]}},
    ]
    pattern_diversity = [
        {"LOWER": "diversity"},
        {"LOWER": "index"},
        {"LOWER": "of"},
        {"LIKE_NUM": True}
    ]
    pattern_governance = [
        {"LOWER": "governance"},
        {"LOWER": "score"},
        {"LOWER": "of"},
        {"LIKE_NUM": True}
    ]
    pattern_landfill = [
        {"LIKE_NUM": True},
        {"TEXT": "%"},
        {"LOWER": "reduction"},
        {"LOWER": "in"},
        {"LOWER": "landfill"},
        {"LOWER": "intensity"}
    ]
    pattern_injury = [
        {"LIKE_NUM": True},
        {"TEXT": "%"},
        {"LOWER": {"IN": ["improvement", "reduction"]}},
        {"LOWER": "in"},
        {"LOWER": "recordable"},
        {"LOWER": "injury"},
        {"LOWER": {"IN": ["frequency", "rate"]}}
    ]
    
    matcher.add("CARBON", [pattern_carbon])
    matcher.add("GHG", [pattern_ghg])
    matcher.add("DIVERSITY", [pattern_diversity])
    matcher.add("GOVERNANCE", [pattern_governance])
    matcher.add("LANDFILL", [pattern_landfill])
    matcher.add("INJURY", [pattern_injury])
    
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        if label == "CARBON":
            for token in span:
                if token.like_num:
                    try:
                        metrics["carbon_emission_reduction"] = int(token.text)
                    except ValueError:
                        metrics["carbon_emission_reduction"] = 0
                    break
        elif label == "GHG":
            for token in span:
                if token.like_num:
                    try:
                        metrics["ghg_emission_reduction"] = int(token.text)
                    except ValueError:
                        metrics["ghg_emission_reduction"] = 0
                    break
        elif label == "DIVERSITY":
            for token in span:
                if token.like_num:
                    try:
                        metrics["diversity_index"] = float(token.text)
                    except ValueError:
                        metrics["diversity_index"] = 0.0
                    break
        elif label == "GOVERNANCE":
            for token in span:
                if token.like_num:
                    try:
                        metrics["governance_score"] = int(token.text)
                    except ValueError:
                        metrics["governance_score"] = 0
                    break
        elif label == "LANDFILL":
            for token in span:
                if token.like_num:
                    try:
                        metrics["landfill_intensity_reduction"] = int(token.text)
                    except ValueError:
                        metrics["landfill_intensity_reduction"] = 0
                    break
        elif label == "INJURY":
            for token in span:
                if token.like_num:
                    try:
                        metrics["recordable_injury_frequency_improvement"] = int(token.text)
                    except ValueError:
                        metrics["recordable_injury_frequency_improvement"] = 0
                    break
    return metrics

####################################
# Financial Metrics Extraction and Health Evaluation
####################################
def extract_financial_metrics(text: str) -> dict:
    """
    Extract key financial metrics from the financial report using spaCy's Matcher.
    Looks for patterns like:
      - "sales and revenues of $67.1B" for revenue
      - "profit per share of $20.12" for profit per share
    Supports numbers with suffixes like 'B' (billion) or 'M' (million).
    """
    metrics = {"revenue": 0, "profit_per_share": 0.0}
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    pattern_revenue = [
        {"LOWER": {"IN": ["sales", "revenues", "revenue"]}},
        {"OP": "*"},
        {"LOWER": "of", "OP": "?"},
        {"IS_CURRENCY": True, "OP": "?"},
        {"LIKE_NUM": True},
        {"TEXT": {"REGEX": "^[BM]$"}, "OP": "?"}
    ]
    matcher.add("REVENUE", [pattern_revenue])

    pattern_profit = [
        {"LOWER": "profit"},
        {"LOWER": "per"},
        {"LOWER": "share"},
        {"LOWER": "of", "OP": "?"},
        {"IS_CURRENCY": True, "OP": "?"},
        {"LIKE_NUM": True}
    ]
    matcher.add("PROFIT", [pattern_profit])

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        label = nlp.vocab.strings[match_id]
        if label == "REVENUE":
            for i, token in enumerate(span):
                if token.like_num:
                    try:
                        number = float(token.text)
                    except ValueError:
                        number = 0
                    unit = None
                    if i + 1 < len(span):
                        next_token = span[i+1]
                        if next_token.text.upper() in ["B", "M"]:
                            unit = next_token.text.upper()
                    if unit == "B":
                        number *= 1e9
                    elif unit == "M":
                        number *= 1e6
                    metrics["revenue"] = number
                    break
        elif label == "PROFIT":
            for token in span:
                if token.like_num:
                    try:
                        metrics["profit_per_share"] = float(token.text)
                    except ValueError:
                        metrics["profit_per_share"] = 0.0
                    break
    return metrics

def evaluate_financial_health(financial_metrics: dict) -> Tuple[bool, List[str]]:
    """
    Evaluate if the company is financially sound.
    Checks if the company is loss-making (negative profit per share) or if revenue is zero.
    Returns a tuple (is_healthy: bool, reasons: list).
    """
    reasons = []
    is_healthy = True
    if financial_metrics.get("profit_per_share", 0) < 0:
        is_healthy = False
        reasons.append("Company is loss-making (negative profit per share).")
    if financial_metrics.get("revenue", 0) == 0:
        is_healthy = False
        reasons.append("Financial report indicates zero revenue.")
    return is_healthy, reasons

####################################
# Exclusion Criteria and Overall Evaluation
####################################
def check_exclusion_criteria(text: str) -> List[str]:
    """
    Check the exclusion criteria from DZ BANK to flag disqualifying activities.
    Keywords and phrases below are derived from the provided exclusion criteria PDF.
    """
    reasons = []
    exclusion_keywords = {
        "coal-fired power plant": "Involvement in coal-fired power plants is excluded.",
        "thermal coal": "Engagement in thermal coal activities is excluded.",
        "oil extraction": "Involvement in oil extraction (e.g., fracking, Arctic drilling) is excluded.",
        "nuclear power": "Financing nuclear power plant activities is excluded.",
        "controversial weapon": "Involvement in controversial weapons is excluded.",
        "arms trade": "Financing arms transactions is excluded.",
        "child labor": "Evidence of child labor is excluded.",
        "pornography": "Financing companies in the pornography industry is excluded.",
        "gambling": "Engagement in controversial forms of gambling is excluded.",
        "conflict material": "Trading raw materials from conflict areas is excluded.",
        "illegal deforestation": "Illegal deforestation practices are excluded.",
        "deforestation": "Deforestation without sustainable practices is excluded.",
        "non-compliance": "Non-compliance with regulations is excluded.",
        "sanction": "Subject to international sanctions is excluded."
    }
    for keyword, reason in exclusion_keywords.items():
        if re.search(keyword, text, re.IGNORECASE):
            reasons.append(reason)
    return reasons

def evaluate_organization(esg_text: str, financial_text: str, exclusion_text: str) -> dict:
    """
    Evaluate the organization by processing ESG, financial, and exclusion reports.
    Generates detailed reasons and assigns a loan option if approved.
    """
    esg_metrics = extract_esg_metrics(esg_text)
    financial_metrics = extract_financial_metrics(financial_text)
    exclusion_reasons = check_exclusion_criteria(exclusion_text)
    
    evaluation = {
        "esg_metrics": esg_metrics,
        "financial_metrics": financial_metrics,
        "exclusion_reasons": exclusion_reasons,
        "approved": False,
        "loan_option": None,
        "reasons": []
    }
    
    # Immediately reject if any exclusion criteria are met.
    if exclusion_reasons:
        evaluation["reasons"].extend(exclusion_reasons)
        evaluation["reasons"].append("Organization meets exclusion criteria and is rejected.")
        return evaluation
    
    # Evaluate ESG metrics against thresholds.
    esg_approved = True
    if esg_metrics['carbon_emission_reduction'] < ESG_CRITERIA['carbon_emission_reduction']:
        esg_approved = False
        evaluation["reasons"].append(
            f"Carbon emission reduction is {esg_metrics['carbon_emission_reduction']}%, below the required {ESG_CRITERIA['carbon_emission_reduction']}%."
        )
    else:
        evaluation["reasons"].append(
            f"Carbon emission reduction is {esg_metrics['carbon_emission_reduction']}%, meeting the required threshold."
        )
    if esg_metrics['diversity_index'] < ESG_CRITERIA['diversity_index']:
        esg_approved = False
        evaluation["reasons"].append(
            f"Diversity index is {esg_metrics['diversity_index']}, below the required {ESG_CRITERIA['diversity_index']}."
        )
    else:
        evaluation["reasons"].append(
            f"Diversity index is {esg_metrics['diversity_index']} meeting the required threshold."
        )
    if esg_metrics['governance_score'] < ESG_CRITERIA['governance_score']:
        esg_approved = False
        evaluation["reasons"].append(
            f"Governance score is {esg_metrics['governance_score']}, below the required {ESG_CRITERIA['governance_score']}."
        )
    else:
        evaluation["reasons"].append(
            f"Governance score is {esg_metrics['governance_score']} meeting the required threshold."
        )
    
    # Evaluate financial health.
    healthy, fin_reasons = evaluate_financial_health(financial_metrics)
    if not healthy:
        esg_approved = False
        evaluation["reasons"].extend(fin_reasons)
    
    # Determine overall approval and select a loan option based on performance.
    if esg_approved:
        evaluation["approved"] = True
        if esg_metrics['carbon_emission_reduction'] >= ESG_CRITERIA['carbon_emission_reduction'] * 1.2:
            evaluation["loan_option"] = "Best-in-Class Investment"
            evaluation["reasons"].append("Outstanding carbon reduction qualifies for Best-in-Class Investment.")
        elif esg_metrics['carbon_emission_reduction'] >= ESG_CRITERIA['carbon_emission_reduction'] * 0.9:
            evaluation["loan_option"] = "Transition Loan"
            evaluation["reasons"].append("Good carbon reduction qualifies for a Transition Loan.")
        else:
            evaluation["loan_option"] = "Thematic Investment"
            evaluation["reasons"].append("Qualifies for a Thematic Investment based on targeted ESG strengths.")
    else:
        evaluation["approved"] = False
        evaluation["loan_option"] = None
        evaluation["reasons"].append("Overall ESG/Financial performance does not meet sustainable loan criteria.")
    
    # Limit reasons to 10 items for clarity.
    evaluation["reasons"] = evaluation["reasons"][:10]
    return evaluation

####################################
# Pydantic Models for API Responses
####################################
class UploadResponse(BaseModel):
    doc_id: str
    organization_id: str
    document_type: str

class EvaluationResponse(BaseModel):
    evaluation_id: str
    organization_id: str
    approved: bool
    loan_option: Optional[str] = None
    summary: str

class DetailedEvaluationResponse(BaseModel):
    evaluation_id: str
    organization_id: str
    approved: bool
    loan_option: Optional[str] = None
    reasons: List[str]
    esg_metrics: dict
    financial_metrics: dict
    exclusion_reasons: List[str]

####################################
# API Endpoints
####################################
@app.post("/api/documents/upload", response_model=UploadResponse)
async def upload_document(
    organization_id: str = Form(...),
    document_type: str = Form(...),
    file: UploadFile = File(...)
):
    if document_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(status_code=400, detail="Invalid document type. Allowed: esg, financial, exclusion.")
    doc_id = save_file(file, organization_id, document_type)
    return UploadResponse(doc_id=doc_id, organization_id=organization_id, document_type=document_type)

@app.get("/api/documents")
async def list_documents(organization_id: str):
    docs = get_documents_by_org(organization_id)
    return {"documents": docs}

@app.post("/api/evaluation", response_model=EvaluationResponse)
async def trigger_evaluation(
    organization_id: str = Form(...),
    esg_doc_id: Optional[str] = Form(None),
    financial_doc_id: Optional[str] = Form(None),
    exclusion_doc_id: Optional[str] = Form(None)
):
    docs = get_documents_by_org(organization_id)
    esg_doc = next((d for d in docs if d["document_type"] == "esg" and (esg_doc_id is None or d["doc_id"] == esg_doc_id)), None)
    fin_doc = next((d for d in docs if d["document_type"] == "financial" and (financial_doc_id is None or d["doc_id"] == financial_doc_id)), None)
    excl_doc = next((d for d in docs if d["document_type"] == "exclusion" and (exclusion_doc_id is None or d["doc_id"] == exclusion_doc_id)), None)

    if not esg_doc or not fin_doc or not excl_doc:
        raise HTTPException(status_code=400, detail="Missing one or more required document types for evaluation.")

    # Read document contents
    esg_text = read_document(esg_doc["doc_id"])
    fin_text = read_document(fin_doc["doc_id"])
    excl_text = read_document(excl_doc["doc_id"])

    # Evaluate organization based on documents
    evaluation = evaluate_organization(esg_text, fin_text, excl_text)
    evaluation_id = str(uuid.uuid4())
    evaluation["evaluation_id"] = evaluation_id
    evaluation["organization_id"] = organization_id
    evaluations_db[evaluation_id] = evaluation

    summary = "Approved" if evaluation["approved"] else "Rejected"
    return EvaluationResponse(
        evaluation_id=evaluation_id,
        organization_id=organization_id,
        approved=evaluation["approved"],
        loan_option=evaluation["loan_option"],
        summary=summary
    )

@app.get("/api/evaluation/{evaluation_id}", response_model=DetailedEvaluationResponse)
async def get_evaluation(evaluation_id: str):
    evaluation = evaluations_db.get(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return DetailedEvaluationResponse(
        evaluation_id=evaluation_id,
        organization_id=evaluation["organization_id"],
        approved=evaluation["approved"],
        loan_option=evaluation["loan_option"],
        reasons=evaluation["reasons"],
        esg_metrics=evaluation["esg_metrics"],
        financial_metrics=evaluation["financial_metrics"],
        exclusion_reasons=evaluation["exclusion_reasons"]
    )

@app.get("/api/evaluation/{evaluation_id}/reasons")
async def get_evaluation_reasons(evaluation_id: str):
    evaluation = evaluations_db.get(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"evaluation_id": evaluation_id, "reasons": evaluation["reasons"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

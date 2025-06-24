import base64
import json
import re
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List
from fuzzywuzzy import fuzz
from mistralai import Mistral
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Bank Statement Analyzer API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Categories for transaction classification
CATEGORIES = {
    "loan": ["loan repayment", "emi", "instalment", "nbfc", "propelld", "finserv", "credit card payment"],
    "salary": ["salary", "payroll", "monthly pay", "income", "credited by employer", "neft-salary"],
    "cash": ["cash withdrawal", "atm", "cash deposit", "cash"],
    "irregular": ["reversal", "charge", "penalty", "fee", "unknown", "misc"]
}

def extract_json_objects(text):
    """
    Extract JSON objects from text that might contain multiple JSON objects
    mixed with other content.
    """
    # Find all text between curly braces that could be JSON objects
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    json_candidates = re.findall(json_pattern, text)
    
    # Try to parse each candidate as JSON
    json_objects = []
    for candidate in json_candidates:
        try:
            json_obj = json.loads(candidate)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            # If nested JSON objects cause issues, try a more aggressive approach
            try:
                # Try to find complete JSON objects
                if "account_details" in candidate or "transactions" in candidate:
                    # Clean and fix potential formatting issues
                    cleaned = candidate.replace('\n', ' ').strip()
                    json_obj = json.loads(cleaned)
                    json_objects.append(json_obj)
            except:
                pass
    
    return json_objects

def process_json_data(input_text):
    """
    Process JSON data from input text and extract account details and transactions.
    Preserves original field names exactly as they appear in the source data.
    """
    # Extract JSON objects from the input text
    json_objects = extract_json_objects(input_text)
    
    # Initialize result containers
    account_details = None
    transactions = None
    
    # Classify and store each JSON object
    for obj in json_objects:
        if 'account_details' in obj:
            account_details = obj['account_details']
        elif 'transactions' in obj:
            transactions = obj['transactions']
    
    # Format for output
    results = {
        'account_details': account_details,
        'transactions': transactions
    }
    
    return results

def categorize_transaction(row, description_field="particulars"):
    """
    Categorize a transaction based on its description using fuzzy matching
    """
    description = str(row.get(description_field, "")).lower()
    best_match = None
    best_score = 0
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            score = fuzz.partial_ratio(description, keyword)
            if score > best_score:
                best_score = score
                best_match = category
    return best_match if best_score > 70 else "other"

def clean_transaction_amounts(transactions):
    """
    Cleans numeric fields like credit, debit, withdrawal, deposit, and balance.
    - Removes commas
    - Extracts numeric part from strings like '1,234.56 CR' or '2,000.00 DR'
    - Keeps float values, sets to None if missing or invalid
    """
    amount_fields = ["credit", "debit", "withdrawal", "deposit", "balance"]
    cleaned_transactions = []
    for txn in transactions:
        cleaned_txn = txn.copy()
        for field in txn:
            field_lower = field.lower()
            if any(amt in field_lower for amt in amount_fields):
                value = str(txn[field]).strip().upper()
                # Remove commas, Rs., CR/DR indicators
                value = value.replace(',', '')
                value = re.sub(r"(INR|RS\\.?|CR|DR)", "", value)
                # Extract numeric part
                match = re.search(r"[-+]?\d*\.?\d+", value)
                if match:
                    try:
                        cleaned_txn[field] = float(match.group())
                    except ValueError:
                        cleaned_txn[field] = None
                else:
                    cleaned_txn[field] = None
        cleaned_transactions.append(cleaned_txn)
    
    return cleaned_transactions

def analyze_transactions(transactions):
    """
    Analyze and categorize transactions
    """
    transactions = clean_transaction_amounts(transactions)
    df = pd.DataFrame(transactions)
    
    # Find description column
    description_column = next((col for col in df.columns if col.lower() in ["description", "narration", "particulars"]), None)
    if not description_column:
        # If no standard description column found, use the first text column
        text_cols = [col for col in df.columns if isinstance(df[col].iloc[0] if not df.empty else "", str)]
        description_column = text_cols[0] if text_cols else df.columns[0]
    
    # Categorize transactions
    df["category"] = df.apply(categorize_transaction, axis=1, description_field=description_column)
    
    # Generate summary statistics
    summary = {
        "total_transactions": len(df),
        "total_inflow": df.get("credit", df.get("deposit", pd.Series([0]))).sum(),
        "total_outflow": df.get("debit", df.get("withdrawal", pd.Series([0]))).sum(),
        "category_counts": df["category"].value_counts().to_dict()
    }
    
    # Convert categorized data to dictionary format
    categorized_data = {
        category: df[df["category"] == category].to_dict(orient="records")
        for category in set(df["category"].tolist())
    }
    
    return {
        "transactions": transactions,
        "categorized_transactions": categorized_data,
        "summary": summary
    }

@app.post("/analyze")
async def analyze_bank_statement(
    file: UploadFile = File(...),
    api_key: str = Form(None)
):
    """
    Analyze a bank statement image and return structured data with categorized transactions
    """
    # Validate API key if provided, otherwise use environment variable
    mistral_api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise HTTPException(status_code=400, detail="Mistral API key is required")
    
    # Read and encode the uploaded file
    file_content = await file.read()
    base64_image = base64.b64encode(file_content).decode('utf-8')
    
    # Initialize Mistral client
    # try:
    client = Mistral(api_key=mistral_api_key)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to initialize Mistral client: {str(e)}")
    
    # Create the OCR request
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        """You are an OCR assistant. Extract two separate JSON objects from the image:

1. The first JSON should be 'account_details' with fields matching exactly what appears in the document header (such as name, account_number, ifsc_code, branch, customer_id, etc.)

2. The second JSON should be 'transactions', a list of transaction objects containing fields exactly as they appear in the transaction table columns (such as date, particulars/description/narration, debit/withdrawal, credit/deposit, balance, etc.) while maintaining strict column alignment throughout the ENTIRE document. Pay careful attention to:
   - Maintain the exact same column mapping for ALL rows (not just the first few)
   - Never swap debit/withdrawal and credit/deposit values
   - Ensure each value goes into its correct field based on the column position, not assumptions about the transaction
   - If a column is empty for a particular row, represent it as null or empty string

Output these two JSONs clearly one after the other. Do not wrap them in any other structure or markdown. Just print raw JSONs, one after another. Triple-check that debit/credit values are correctly assigned throughout ALL transaction rows."""
                    )
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ],
        }
    ]
    
    # Send request to Mistral API
    # try:
    chat_response = client.chat.complete(
        model="pixtral-large-latest",
        messages=messages
    )
    response_text = chat_response.choices[0].message.content
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    
    # Process the OCR results
    # try:
    results = process_json_data(response_text)
    
    if not results['transactions']:
        raise HTTPException(status_code=422, detail="No transaction data could be extracted from the image")
    
    # Analyze the transactions
    analysis_results = analyze_transactions(results['transactions'])
    
    # Combine all results
    final_results = {
        "account_details": results['account_details'],
        "transactions": results['transactions'],
        "categorized_transactions": analysis_results['categorized_transactions'],
        "summary": analysis_results['summary']
    }

    print(final_results)
        
    return JSONResponse(content=final_results)
    
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Failed to process data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
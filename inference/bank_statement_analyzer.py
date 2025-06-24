from pdf2image import convert_from_path
from PIL import Image
import base64
import io
import openai
# from openai import OpenAI
import re
import json

import os

openai.api_key = os.getenv("OPENAI_API_KEY")


def pdf_to_images(pdf_path, dpi=300):
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        # poppler_path=r"C:\Users\msimc\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"  # update if needed
    )

def image_to_base64(image: Image.Image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_json_from_response(text_response):
    """
    Cleans up markdown-wrapped JSON and parses it.
    """
    json_str = re.sub(r"^```json|```$", "", text_response.strip(), flags=re.MULTILINE).strip()
    return json.loads(json_str)

def analyze_bank_statement_from_images(images):
    vision_inputs = []
    for image in images:
        img_b64 = image_to_base64(image)
        vision_inputs.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }
        })

    prompt = """
You are an intelligent financial document parser. From the bank statement images, extract:
- Account holder name, account number, bank name
- All transactions with date, description, debit, credit, balance
- Identify salary credits and loan-related transactions
- Flag cheque bounces, penalties, or defaults
- Calculate average monthly balance

Respond only in clean JSON format.
"""

    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [{"type": "text", "text": prompt}] + vision_inputs
    }],
    max_tokens=4000
)

    raw_text = response.choices[0].message.content
    parsed_data = extract_json_from_response(raw_text)

    # Normalize output structure
    structured_output = {
        "account": {
            "holder": parsed_data.get("account_holder"),
            "number": parsed_data.get("account_number"),
            "bank": parsed_data.get("bank_name"),
        },
        "transactions": parsed_data.get("transactions", []),
        "salary_transactions": parsed_data.get("salary_credits", []),
        "loan_transactions": parsed_data.get("loan_related_transactions", []),
        "default_flags": parsed_data.get("flagged_transactions", []),
        "average_balance": parsed_data.get("average_monthly_balance"),
    }

    return structured_output

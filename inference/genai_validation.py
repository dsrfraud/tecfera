import openai
import os
from dotenv import load_dotenv
from fastapi import HTTPException

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

def extract_passport_front_data(front_ocr_text, mrz_text):
    """
    Extracts and validates passport details using GPT, cross-checking the front page and MRZ.
    Returns a structured dictionary with passport details.
    """
    
    prompt = f"""
    You are an expert in extracting and validating passport information from OCR data. 
    Given two OCR text inputs:
    1. "Front Page OCR": contains passport details but may have errors due to OCR inaccuracies.
    2. "MRZ OCR": contains the machine-readable zone (MRZ), which is structured and more reliable.

    Your task:
    - Extract key details from both the OCR inputs: passport number, surname, given name, date of birth, 
      date of issue, date of expiry, place of issue, place of birth, and MRZ.
    - Cross-check these details with the MRZ to validate and correct potential errors in OCR data.
    - Fix any MRZ issues based on the following standard MRZ format:
      1. The MRZ should consist of 44 characters in total, split into two lines (22 characters each).
      2. The MRZ line starts with the passport number, followed by a check character.
      3. The surname and given name appear in the second part of the MRZ, followed by a birth date and expiry date.
      4. The MRZ should only contain letters, numbers, and the '<' symbol.
      5. If there are errors in the MRZ (e.g., missing or misformatted characters), correct them based on other extracted information.
      6. Do not add additional text expect the dict values in response.
    - Ensure the MRZ adheres to the correct format, fixing any mistakes in the extraction from the OCR.
    
    

    ### Front Page OCR:
    {front_ocr_text}

    ### MRZ OCR:
    {mrz_text}

    Tips:
    - If the MRZ has extra characters or formatting issues, remove or correct them.
    - Ensure the dates are in the proper format (DD/MM/YYYY).
    - If the surname or given name is split in the MRZ, fix it by combining the names properly.
    - The MRZ should be corrected to match the format exactly: 44 characters with , separation after 22 letters in each.
    - Verify between surname and given names should have double (<<).
    -- Take reference from the below: 
                    1st line P<ISOCOUNTRYSURNAME<<GIVENNAMES<<<<<<<<<<<<<<
                    2nd line : PASSPORTNUMBER<CHECKDIGITCOUNTRYDOB<CHECKDIGITSEXEXPIRY<CHECKDIGITPERSONALNUMBER<CHECKSUM
    -- For an example : wrong - Z7927844<81ND9101210M3407242E079918828024<48 it should be Z7927844<8IND9101210M3407242E079918828024<48
    -- Strictly only return the dict, do not add any additional text in response.
    -- Also do not generate any code.

     Your response should contain ONLY the following JSON object, with NO additional explanation or text:


    {{
        "passport_number": "<validated_passport_number>",
        "surname": "<validated_surname>",
        "given_name": "<validated_given_name>",
        "date_of_birth": "<DD/MM/YYYY>",
        "date_of_issue": "<DD/MM/YYYY>",
        "date_of_expiry": "<DD/MM/YYYY>",
        "issue_place": "<validated_issue_place>",
        "birth_place": "<validated_birth_place>",
        "mrz": "<44_letters_with_comma_separation_corrected_mrz>"
    }}
    
    If any data is missing or cannot be extracted, return 'no' instead. Do not any extra text before and after dict brackets.
    """
    
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You extract structured passport details and validate them with MRZ data."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )

        if completion:
            extracted_data = completion["choices"][0]["message"]["content"]
            # extracted_data =  extracted_data.replace('json', '').strip()
            
            # Parse the response into a dictionary (assuming it is valid JSON)
            try:
                # extracted_data_dict = json.loads(extracted_data)
                print(extracted_data)
                return extracted_data
            except Exception as e:
                print(f"Error: Failed to decode JSON response: {e}")
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    


def extract_passport_back_data(back_ocr_text):
    
    prompt = f"""
    You are an expert in extracting and validating passport information from OCR data. 
    Given Passport back text inputs:
    1. "Back Page OCR": contains additional passport details like Legal Guardian, name of mother, address, file number.

    ### BACK Page OCR:
    {back_ocr_text}

    Your task:
    - Extract key details from both the OCR inputs: name of legal guardian, name of mother, address, 
      file number.
    - Cross-check these details with the file number to validate and correct potential errors in OCR data.
    - Your intelligency to fix the address with correct information.
    
  

    Your response should contain ONLY the following JSON object, with NO additional explanation or text:

    {{
        "passport_number": "<validated_passport_number>",
        "legal_guardian": "<validated_legal_guardian>",
        "mother_name": "<validated_mother_name>",
        "address": "<validated_address>",
        "file_number": "<validated_file_number>",
    }}
    
    If any data is missing or cannot be extracted, return 'no' instead. Do not any extra text before and after dict brackets.
    """
    
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You extract and validate passport data using back page OCR and MRZ OCR."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )

        if completion:
            extracted_data = completion["choices"][0]["message"]["content"]
            
            try:
                return extracted_data
            except Exception as e:
                print(f"Error: Failed to decode JSON response: {e}")
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None

import re
from fuzzywuzzy import fuzz

def verify_passport_details(data):
    def calculate_checksum(date):
        weights = [7, 3, 1]
        total = sum(int(date[i]) * weights[i % 3] for i in range(len(date)))
        return str(total % 10)
    
    def find_date_with_checksum(text, date):
        formatted_date = date[8:10] + date[3:5] + date[0:2]  # Convert DD/MM/YYYY to YYMMDD
        match = re.search(f"{formatted_date}(\d)", text)
        if match:
            date_checksum = calculate_checksum(formatted_date)
            return "Match" if match.group(1) == date_checksum else "Checksum Mismatch"
        return "Not a Match"
    
    def check_fields(text, fields):
        results = {}
        for field_name, field_value in fields.items():
            match_percentage = fuzz.partial_ratio(field_value.lower(), text.lower())
            max_length = max(len(field_value), 1)  # Avoid division by zero
            allowed_errors = 1
            threshold = ((max_length - allowed_errors) / max_length) * 100  # Dynamic threshold
            results[field_name] = "Match" if match_percentage > threshold else f"Not a Match ({match_percentage}%)"
        return results
    
    mrz_text = data["mrz"]
    fields_to_search = {
        "Surname": data["surname"],
        "First Name": data["given_name"],
        "Passport Number": data["passport_number"]
    }
    
    matches = check_fields(mrz_text, fields_to_search)
    matches["Date of Birth"] = find_date_with_checksum(mrz_text, data["date_of_birth"])
    matches["Date of Expiry"] = find_date_with_checksum(mrz_text, data["date_of_expiry"])

    print(f'*********************{matches}')
    
    return matches

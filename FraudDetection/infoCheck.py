# Information Verification
import pandas as pd
from datetime import date as Date
from pathlib import Path

current_dir = Path(__file__).parent
location_code_df_path = current_dir / 'LocationCode.csv'

location_code_df = pd.read_csv(location_code_df_path)

def check_id_number(id_number):
    id_number = id_number.replace(" ", "")
    flags = {
        "id_length_valid": False,
        "id_location_code_valid": False,
        "id_checksum_valid": False,
    }

    # Check length
    if len(id_number) == 13 and id_number.isdigit():
        flags["id_length_valid"] = True
    else:
        return flags
    
    # Check location code
    location_code_extract = int(id_number[1:5])
    if location_code_extract in location_code_df['LocationCode'].values:
        flags["id_location_code_valid"] = True
        print(f"Location Match: {location_code_df[location_code_df['LocationCode'] == location_code_extract]['RegistrarOffice'].values[0]}")


    digits = [int(d) for d in id_number]
    checksum = sum(d * (13 - i) for i, d in enumerate(digits[:-1])) % 11
    checksum_digit = (11 - checksum) % 10
    if checksum_digit == digits[-1]:
        flags["id_checksum_valid"] = True

    
    return flags

def check_request_number(request_number, issue_date):
    request_number = request_number.replace(" ", "")
    flags = {
        "request_length_valid": False,
        "request_format_valid": False,
        "request_location_code_valid": False,
        "issue_date_valid": False,
    }

    # Check length
    if len(request_number) == 16:
        flags["request_length_valid"] = True
    else:
        return flags
    
    # Check format
    request_number_no_dash = request_number.replace('-', '')
    if request_number[4] == '-' and request_number[7] == '-' and request_number_no_dash.isdigit():
        flags["request_format_valid"] = True
    else:
        return flags
    
    # Check location code
    location_code_extract = int(request_number[0:4])
    if location_code_extract in location_code_df['LocationCode'].values:
        flags["request_location_code_valid"] = True

    # Check issue date
    issue_date_split = issue_date.split('-')
    if len(issue_date_split) != 3:
        return flags
    for i in range(2):
        if issue_date_split[i].isdigit() == False:
            return flags
    date_str = request_number[8:12]
    if date_str.isdigit() == False:
        return flags
    month = int(date_str[0:2])
    date = int(date_str[2:4])
    print(f"Extracted Date: {date}, Month: {month}, Issue Date: {issue_date_split}")
    if month == int(issue_date_split[1]) and date == int(issue_date_split[2]):
        flags["issue_date_valid"] = True
    
    return flags

def run_info_check(data):
    # data["issueDate"] = date(2020,1,1)
    data.update(check_id_number(data["id"]))
    # data.update(check_request_number(data["numUnderImage"], Date(2020,1,1)))
    data.update(check_request_number(data["requestNumber"], data['Issue']))
    return data

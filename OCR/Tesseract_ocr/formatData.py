import re
from pathlib import Path
current_dir = Path(__file__).parent

def split_thai_name(full_name):
	# Military and Police Abbreviations
    abbreviations = r'(?:[ก-ฮ]{1,3}\.)+(?:หญิง|ญ\.)?'
    
    # Full Ranks
    full_ranks = r'(?:พลตำรวจ|พันตำรวจ|ร้อยตำรวจ|จ่าสิบ|สิบ|ว่าที่ร้อย|นาวาอากาศ|เรืออากาศ|นาวา|เรือ|พล|พัน|ร้อย|จ่า|สิบ|หมู่)(?:เอก|โท|ตรี|จัตวา|อากาศ|เรือ|ตำรวจ)?(?:พิเศษ)?(?:หญิง)?(?!\.)'
    
    # Others
    titles = r'(?:นาย|นางสาว|นาง|น\.ส\.|คุณ|ท่าน|ดร\.|นพ\.|พญ\.|พระ|สามเณร|หม่อมหลวง|ม\.ล\.|หม่อมราชวงศ์|ม\.ร\.ว\.|ศาสตราจารย์|รองศาสตราจารย์|ผู้ช่วยศาสตราจารย์|ศ\.|รศ\.|ผศ\.)'

    master_pattern = f"^({abbreviations}|{full_ranks}|{titles})"
    
    full_name = full_name.strip()
    
    remaining = full_name
    
    while True:
        match = re.match(master_pattern, remaining)
        if match:
            remaining = remaining[len(match.group(0)):].strip()
        else:
            break

    parts = remaining.split()
    first_name = ""
    middle_name = ""
    last_name = ""

    if len(parts) > 0:
        first_name = parts[0]
        
    if len(parts) == 2:
        last_name = parts[1]
    elif len(parts) > 2:
        middle_name = parts[1]
        last_name = " ".join(parts[2:])
        
    parts = remaining.split()
    first_name = ""
    middle_name = ""
    last_name = ""

    if len(parts) > 0:
        first_name = parts[0]
        
    if len(parts) == 2:
        last_name = parts[1]
    elif len(parts) > 2:
        middle_name = parts[1]
        last_name = " ".join(parts[2:])
        
    first_name = ' '.join([first_name, middle_name]).strip()
    return first_name, last_name
def format_date (date, connectBy, lang) :
    if date == "ตลอดชีพ" or date == "LIFELONG" :
        return date
    date_parts = date.split('/')
    if lang == "eng" :
        result = connectBy.join([date_parts[2], date_parts[1], date_parts[0]])
    else :
        result = connectBy.join([date_parts[0], date_parts[1], date_parts[2]])
    return result
def process_extracted_data(data):
    data['idCardNo'] = data['id']
    first_name_th, last_name_th = split_thai_name(data['fullnameTh_raw'])
    data["fullnameTh"] = data['fullnameTh_raw']
    data['firstNameTh'] = first_name_th
    data['lastNameTh'] = last_name_th
    data['fullnameEn'] = data['firstnameEn_raw'] + ' ' + data['lastNameEn']
    parts = data['firstnameEn_raw'].split()
    if len(parts) > 0:
        data['firstNameEn'] = ' '.join(parts[1:])
    data['Birth'] = format_date(data['birthDateEN'], '-', 'eng')
    data['birthDateTh'] = format_date(data['birthDateTH_crossLangFix'], ' ', 'tha')
    data['Issue'] = format_date(data['issueDateEN'], '-', 'eng')
    data['issueDateTh'] = format_date(data['issueDateTH_crossLangFix'], ' ', 'tha')
    data['Expire'] = format_date(data['expireDateEN'], '-', 'eng')
    data['expiryDateTh'] = format_date(data['expireDateTH_crossLangFix'], ' ', 'tha')
    
	# data['Birth'] = data['birthDateEN']
    # data['birthDateTh'] = data['birthDateTH_crossLangFix']

    # issueDate_parts = data['issueDate'].split('-')
    # if len(issueDate_parts) == 3:
    #     day = issueDate_parts[2]
    #     month = issueDate_parts[1]
    #     year = int(issueDate_parts[0]) - 543
    #     data['Issue'] = f"{year}-{month}-{day}"
    # data['issueDateTh'] = data['issueDate_alpha']

    # expireDate_parts = data['expireDate'].split('-')
    # if len(expireDate_parts) == 3:
    #     day = expireDate_parts[2]
    #     month = expireDate_parts[1]
    #     year = int(expireDate_parts[0]) - 543
    #     data['Expire'] = f"{year}-{month}-{day}"
    # data['expiryDateTh'] = data['expireDate_alpha']

    # # addressFull,addressNo,moo,trok,soi,street,subDistrict,district,province
    # data['addressFull'] = data['address_raw']
    return data
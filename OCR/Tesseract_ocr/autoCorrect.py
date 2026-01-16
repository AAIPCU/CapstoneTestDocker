from pathlib import Path
from datetime import datetime
import re

import cv2
import pandas as pd
import pytesseract
from rapidfuzz import fuzz

current_dir = Path(__file__).parent
excel_path = current_dir / 'ThepExcel-Thailand-Tambon.xlsx'

monthTHList = ['ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.', '-', 'ตลอดชีพ']
monthEngList = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May.', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.', '-', 'LIFELONG']

def _lookup_score(value, dictionary):
	labels, scores = dictionary
	return dict(zip(labels, scores)).get(value, 0.0)

def _safe_month_index(value, candidates):
	if not value:
		return None
	try:
		return candidates.index(value)
	except ValueError:
		return None

def _safe_int(value):
	try:
		return int(value)
	except (TypeError, ValueError):
		return None

def _prepare_tokens(value):
	if isinstance(value, (list, tuple)):
		return [str(token) for token in value if token]
	return [str(value)] if value else []

def _extract_components(tokens):
	day_chars, month_chars, year_chars = [], [], []
	has_day = False
	has_month = False
	co = 0
	for token in tokens:
		for ch in token:
			co += 1
			if ch.isdigit() or (ch == '-' and not has_day):
				target = year_chars if has_month else day_chars
				target.append(ch)
				has_day = True
			elif ch == ' ' and not has_day:
				has_day = True
			elif (has_day and ch !=' ') or (co > 2):
				month_chars.append(ch)
				has_month = True
	return ''.join(day_chars), ''.join(month_chars) or ''.join(tokens), ''.join(year_chars)

def _resolve_month_index(name, primary, alternate):
	if not name:
		return None
	index = _safe_month_index(name, primary)
	return index if index is not None else _safe_month_index(name, alternate)

def _format_dual_date(day, month_alpha, month_index, year, *, is_thai_numeric=False):
	day_text = '' if day in ('', '-') else day
	day_numeric = day_text.zfill(2) if day_text  else ''
	month_text = '' if month_alpha in ('', '-') else month_alpha
	month_numeric = f"{month_index + 1:02d}" if month_index is not None and month_text else ''
	year_text = '' if year in ('', '-', None) else str(year)
	numeric_year = year_text
	# if numeric_year and is_thai_numeric:
	# 	year_val = _safe_int(numeric_year)
	# 	if year_val is not None and year_val < 2500:
	# 		numeric_year = str(year_val + 543)
	return '/'.join([day_text, month_text, year_text]), '/'.join([day_numeric, month_numeric, numeric_year])

def _choose_month_index(idx_th, idx_en, th_score, en_score):
	if idx_th is None:
		return idx_en
	if idx_en is None:
		return idx_th
	return idx_th if th_score > en_score else idx_en

def _choose_year(year_th, year_en, dictTH, dictEN):
	year_en_val = _safe_int(year_en)
	year_th_val = _safe_int(year_th)
	if year_en_val is None and year_th_val is None:
		return None
	year_th_gregorian = year_th_val - 543 if year_th_val is not None else None
	if year_en_val is None:
		return year_th_gregorian
	if year_th_gregorian is None or year_en_val == year_th_gregorian:
		return year_en_val
	score_th = _lookup_score(year_th, dictTH)
	score_en = _lookup_score(year_en, dictEN)
	current_year = datetime.now().year
	valid = lambda y: abs(y - current_year) <= 300
	if score_en >= score_th:
		return year_en_val if valid(year_en_val) else (year_th_gregorian if valid(year_th_gregorian) else None)
	return year_th_gregorian if valid(year_th_gregorian) else (year_en_val if valid(year_en_val) else None)

def replace_noise_text(text):
	if text is None:
		return ''
	cleaned = str(text).replace('!', '').replace('\n', '').replace('\x0c', '')
	return cleaned.strip().replace(' ', '')

def fuzzy_find(keyword, text, threshold=0.8):
	keyword = keyword or ''
	text = text or ''
	if not keyword or not text:
		return None, 0.0, -1
	k = len(keyword)
	min_score = threshold * 100
	best_score = min_score
	best_sub = None
	best_index = -1
	if len(text) < k:
		text = text.ljust(k)
	for i in range(max(len(text) - k + 1, 0)):
		sub = text[i:i + k]
		score = fuzz.ratio(keyword, sub)
		if score > best_score:
			best_score = score
			best_sub = sub
			best_index = i
	if best_sub is None:
		return None, 0.0, -1
	return best_sub, best_score / 100.0, best_index

def _best_from_candidates(text, candidates, threshold=0.8, prefixer=None):
	if isinstance(text, (list, tuple)):
		search_text = ''.join(str(t) for t in text)
	else:
		search_text = str(text or '')
	if not search_text:
		return None, None, 0.0, -1
	best_cand = None
	best_match = None
	best_sim = threshold
	best_index = -1
	for cand in candidates:
		query = prefixer(cand) if prefixer else cand
		if not query:
			continue
		match, sim, index = fuzzy_find(query, search_text, threshold)
		if match and sim > best_sim:
			best_cand = cand
			best_match = match
			best_sim = sim
			best_index = index
	if best_cand is None:
		return None, None, 0.0, -1
	return best_cand, best_match, best_sim, best_index

def replace_keyword(text, keyword):
	if not text:
		return text
	best_match, _, best_index = fuzzy_find(keyword, text, threshold=0.51)
	if best_match is None or best_index < 0:
		return text
	def _is_edge_char(ch):
		return ch.isdigit() or ch in '/-'
	leading_len = 0
	while leading_len < len(best_match) and _is_edge_char(best_match[leading_len]):
		leading_len += 1
	trailing_len = 0
	while trailing_len < len(best_match) - leading_len and _is_edge_char(best_match[-(trailing_len + 1)]):
		trailing_len += 1
	leading = best_match[:leading_len]
	trailing = best_match[len(best_match) - trailing_len:] if trailing_len else ''
	replacement = leading + keyword + trailing
	replaced = text[:best_index] + replacement + text[best_index + len(best_match):]
	keyword_start = best_index + len(leading)
	keyword_end = keyword_start + len(keyword)
	if keyword_start > 0 and replaced[keyword_start - 1] != ' ':
		replaced = replaced[:keyword_start] + ' ' + replaced[keyword_start:]
		keyword_start += 1
		keyword_end += 1
	if keyword and keyword[-1] != '.' and keyword_end < len(replaced) and replaced[keyword_end] != ' ':
		replaced = replaced[:keyword_end] + ' ' + replaced[keyword_end:]
	return replaced

_ADDRESS_COMPONENT_BOUNDARY = r'(?:หมู่ที่|หมู่|ม\.|ตรอก|ตร\.|ซอย|ซ\.|ถนน|ถ\.|ตำบล|ต\.|อำเภอ|อ\.|จังหวัด|จ\.|$)'

_ADDRESS_COMPONENT_PATTERNS = {
	'moo': re.compile(r'(?:หมู่ที่|หมู่|ม\.)\s*(?P<value>[0-9]{1,4})'),
	'trok': re.compile(r'(?:ตรอก|ตร\.)\s*(?P<value>.+?)(?=' + _ADDRESS_COMPONENT_BOUNDARY + ')'),
	'soi': re.compile(r'(?:ซอย|ซ\.)\s*(?P<value>.+?)(?=' + _ADDRESS_COMPONENT_BOUNDARY + ')'),
	'street': re.compile(r'(?:ถนน|ถ\.)\s*(?P<value>.+?)(?=' + _ADDRESS_COMPONENT_BOUNDARY + ')'),
}

def _clean_address_component(value):
	if not value:
		return None
	cleaned = value.strip()
	cleaned = re.sub(r'^[,.;:/\\\s]+', '', cleaned)
	cleaned = re.sub(r'[,.;:/\\\s]+$', '', cleaned)
	return cleaned or None

def _extract_address_components(address_text):
	components = {'addressNo': None, 'moo': None, 'trok': None, 'soi': None, 'street': None}
	if not address_text:
		return components
	for key, pattern in _ADDRESS_COMPONENT_PATTERNS.items():
		match = pattern.search(address_text)
		if not match:
			continue
		value = match.group('value') if 'value' in match.groupdict() else None
		if key == 'moo' and value:
			components[key] = value.strip()
			continue
		cleaned = _clean_address_component(value)
		if cleaned:
			components[key] = cleaned
	return components

def match_location_from_text(text, threshold=0.8):
	df = pd.read_excel(excel_path, sheet_name='AddressDatabase')
	out = {
		'province': None, 'province_match': None, 'province_similarity': 0.0, 'province_index': -1,
		'district': None, 'district_match': None, 'district_similarity': 0.0, 'district_index': -1,
		'tambon': None, 'tambon_match': None, 'tambon_similarity': 0.0, 'tambon_index': -1,
	}
	for level in ('province', 'district', 'tambon'):
		
		if level == 'province':
			candidates = df['ProvinceThai'].unique()
			prefixer = lambda c: c if c == 'กรุงเทพมหานคร' else ('จ.' + c)
		elif level == 'district':
			if not out['province']:
				break
			candidates = df.loc[df['ProvinceThai'] == out['province'], 'DistrictThai'].unique()
			prefixer = (lambda c: 'เขต' + c) if out['province'] == 'กรุงเทพมหานคร' else (lambda c: 'อ.' + c)
		else:
			if not out['district']:
				break
			candidates = df.loc[df['DistrictThai'] == out['district'], 'TambonThai'].unique()
			prefixer = (lambda c: 'แขวง' + c) if out['province'] == 'กรุงเทพมหานคร' else (lambda c: 'ต.' + c)
		best_cand, best_match, best_sim, best_index = _best_from_candidates(text, candidates, threshold, prefixer)
		
		if not best_cand or best_sim < threshold:
			break
		key = level
		out[key] = best_cand
		out[f'{key}_match'] = best_match
		out[f'{key}_similarity'] = best_sim
		out[f'{key}_index'] = best_index
		
	best_match, _, best_index = fuzzy_find('ที่อยู่', text, threshold=0.5)
	search_limit = out['tambon_index'] if out['tambon_index'] >= 0 else len(text)
	address_segment = text[:search_limit]
	if best_match and best_index >= 0:
		address_segment = address_segment[best_index + len(best_match):]
	address_segment = replace_keyword(address_segment, 'หมู่ที่')
	address_segment = replace_keyword(address_segment, 'ซ.')
	address_segment = replace_keyword(address_segment, 'ถ.')
	address_segment = replace_keyword(address_segment, 'ตรอก')
	address = address_segment.strip()
	component_starts = []
	for pattern in _ADDRESS_COMPONENT_PATTERNS.values():
		match = pattern.search(address)
		if match:
			component_starts.append(match.start())
	free_form = address[:min(component_starts)] if component_starts else address
	free_form_clean = _clean_address_component(free_form) or (free_form.strip() or None)
	address_components = _extract_address_components(address)
	address_components['addressNo'] = free_form_clean
	address_components['subDistrict'] = out['tambon']
	address_components['district'] = out['district']
	address_components['province'] = out['province']
	return out, address, address_components

def correct_date(date, lang='tha'):
	tokens = _prepare_tokens(date)
	day, month_raw, year = _extract_components(tokens)
	month_candidate = replace_noise_text(month_raw)
	month_list = monthEngList if lang == 'eng' else monthTHList
	month, _, score, _ = _best_from_candidates(month_candidate, month_list, threshold=0.49)
	if month == 'ตลอดชีพ':
		return 'ตลอดชีพ', 'ตลอดชีพ', score
	elif month == 'LIFELONG':
		return 'LIFELONG', 'LIFELONG', score
	month_name = month or ''
	alt_list = monthEngList if month_list is monthTHList else monthTHList
	month_index = _resolve_month_index(month_name, month_list, alt_list)
	if day not in ('', '-',' ') and int(day) > 31:
		day = ''
	alpha, numeric = _format_dual_date(day, month_name, month_index, year, is_thai_numeric=(lang == 'tha'))
	return alpha, numeric, score

def correct_twodate(th, eng, dictTH, dictEN, monthTHScore, monthENGScore):
	if th == 'ตลอดชีพ' or eng == 'LIFELONG':
		return ('ตลอดชีพ', 'ตลอดชีพ'), ('LIFELONG', 'LIFELONG')
	day_th, month_th, year_th = (th.split('/') + ['', '', ''])[:3]
	day_en, month_en, year_en = (eng.split('/') + ['', '', ''])[:3]
	day = day_en or day_th
	if day_th != day_en and _lookup_score(day_th, dictTH) > _lookup_score(day_en, dictEN):
		day = day_th
	month_idx_th = _safe_month_index(month_th, monthTHList)
	month_idx_en = _safe_month_index(month_en, monthEngList)
	month_index = _choose_month_index(month_idx_th, month_idx_en, monthTHScore, monthENGScore)
	month_th_value = monthTHList[month_index] if month_index is not None and month_index < len(monthTHList) else ''
	month_en_value = monthEngList[month_index] if month_index is not None and month_index < len(monthEngList) else ''
	year_candidate = _choose_year(year_th, year_en, dictTH, dictEN)
	year_en_str = str(year_candidate) if year_candidate is not None else ''
	year_th_str = str(year_candidate + 543) if year_candidate is not None else ''
	th_alpha, th_numeric = _format_dual_date(day, month_th_value, month_index, year_th_str, is_thai_numeric=True)
	en_alpha, en_numeric = _format_dual_date(day, month_en_value, month_index, year_en_str)
	return (th_alpha, th_numeric), (en_alpha, en_numeric)

if __name__ == "__main__":
	image_paths = [
		current_dir / 'test_data' / 'crop3_8.jpg',
		current_dir / 'test_data' / 'crop3_9.jpg',
	]
	ocr_chunks = []
	for path in image_paths:
		img = cv2.imread(str(path))
		if img is None:
			continue
		ocr_chunks.append(
			pytesseract.image_to_string(
				img,
				lang='tha',
				config='--psm 6 -c preserve_interword_spaces=1',
			)
		)
	if not ocr_chunks:
		print('No images available for OCR demo.')
	else:
		text = replace_noise_text(' '.join(ocr_chunks))
		hierarchy, address, summary = match_location_from_text(text, threshold=0.8)
		print('Match result:')
		print(f"  Province: {hierarchy['province']} (match='{hierarchy['province_match']}', sim={hierarchy['province_similarity']:.3f})")
		print(f"  District: {hierarchy['district']} (match='{hierarchy['district_match']}', sim={hierarchy['district_similarity']:.3f})")
		print(f"  Tambon:   {hierarchy['tambon']} (match='{hierarchy['tambon_match']}', sim={hierarchy['tambon_similarity']:.3f})")
		print(f"  Address:  {address}")
		print('Summary:')
		print(f"  Province: {summary['province']}")
		print(f"  District: {summary['district']}")
		print(f"  Tambon:   {summary['tambon']}")
		print(f"  Address:  {summary['address']}")
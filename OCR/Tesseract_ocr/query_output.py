def query_output(data) :
	output = {}
	addressFull = ""
	address_field = {"addressNo" : "", "moo" : "หมู่ที่ ", "trok" : "ตรอก", "soi" : "ซ.", "street" : "ถ.", "subDistrict" : "ต.", "district" : "อ.", "province" : "จ."}
	for item in data :
		if item != "image" :
			for subItem in data[item] :
				itemName = item
				if item == "nameTH" :
					itemName = "fullnameTh"
				elif item == "nameEN" :
					itemName = "firstnameEn"
				elif item == "lastNameEN" :
					itemName = "lastNameEn"
				
				name = itemName+"_"+str(subItem)
				if subItem == "numeric" :
					name = itemName
				elif (item == "lastNameEN" or item == "firstNameEN"or item == "requestNumber" or item == "id") and subItem == "raw" :
					name = itemName
				elif item == "address" and (subItem != "raw" and subItem != "score") :
					name = subItem
					if data[item][subItem] != None and data[item][subItem] != "" :
						addressFull += address_field[subItem] + str(data[item][subItem]) + " "
				elif item != "address" and addressFull != "" :
					output["addressFull"] = addressFull
					addressFull = ""
					
				output[name] = data[item][subItem]
		elif item == "address" :
			for subItem in data[item] :
				itemName = ""
				
				output[itemName+subItem] = data[item][subItem]
	return output

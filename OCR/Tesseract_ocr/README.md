# How to
- ไปโหลด tesseract มาใช้ และโหลดภาษามาสองอันคือไทยกับอังกฤษ (tha.traineddata, eng.traineddata) อนาคตอาจจะเปลี่ยน model
- Final.py ใช้รันทั้ง flow 
- crop_img.py ใช้สำหรับ crop รูปที่ได้มาจาก auto-capture
- extract_data.py ใช้สำหรับทำ OCR และ import autocorrect เข้าไปแล้ว data ที่ได้จะมีแบบแก้ไขแล้วด้วย
- autoCorrect.py ใช้สำหรับแก้ไขวันเกิดและที่อยู่ (ใช้ใน extract_data.py)
- imageQualitySharpness.py ชื่อยาวหน่อยอย่าบ่น ไฟล์มันซ้อนเยอะ อันนี้เอาไว้ทำ imageprocessing แต่ว่าตอนนี้มีปรับแค่ sharpness อย่างอื่นทำละมัน OCR กากขึ้น
- find_locate.py ใช้สำหรับ crop แต่ละ field อยากได้ field ไหนเพิ่มก็ crop เอาจากไฟล์นี้ ได้ออกมาเป็นไฟล์ config.json มันจะเอาไป OCR ต่อ 
- config.json ไฟล์สำหรับเก็บว่า แต่ละ field config อะไรบ้าง มี blacklist อะไร แล้วก็ใช้ model ไหน predict ตำแหน่งบนบัตรอยู่ตรงไหน

file template อยู่ใน folder test_data เลือกเอาอยากใช้ ไฟล์ไหน แต่ว่าถ้าใช้ Raw.png จะต้องไปแก้ config ปรับตำแหน่งของแต่ละ field ใหม่ เพราะว่าอัตราส่วนรูปมันไม่เท่ากัน (อัตราส่วนกว้างยาว)

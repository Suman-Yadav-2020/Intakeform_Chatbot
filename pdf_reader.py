def extract_fields_from_pdf(file_path):
    fields = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                for line in text.split("\n"):
                    if ':' in line:
                        fields.append(line.strip())
    return fields
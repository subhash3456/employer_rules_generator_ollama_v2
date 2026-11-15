
import fitz

pdf_path = "C:\\Users\\subha\\Downloads\\text_to_pdf_Imagetotext (1).pdf"

doc = fitz.open(pdf_path)

print("Total pages:", len(doc))

for index, page in enumerate(doc, start=1):
    print(f"\n========== PAGE {index} ==========")

    text = page.get_text("text")

    print("Extracted characters:", len(text))
    print(text[:1500])

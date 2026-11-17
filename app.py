from pathlib import Path
import json

from src.document_loader import load_pdf
from src.text_processing import clean_txt

BASE_DIR = Path(__file__).resolve().parent

def main():
    pdf_name = input("Enter PDF name: ")
    pdf_path = BASE_DIR / "data" / "input" / pdf_name
    
    pages = load_pdf(pdf_path)
    
    processed_page = []
    
    for p in pages:
        cleaned_text = clean_txt(p["content"])
        
        if cleaned_text:
            processed_page.append({
                "page_number":p["page_number"],
                "content":cleaned_text
            })
    
    output_path = BASE_DIR / "data" / "output" / f"{Path(pdf_name).stem}.json"
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(processed_page, file, indent=4, ensure_ascii=False)
    
    print("Json File Created Successfully")

if __name__ == "__main__":
    main()
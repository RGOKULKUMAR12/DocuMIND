import pymupdf as pm
from pathlib import Path

def load_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are allowed")
    
    doc = pm.open(pdf_path)
    
    pages = []
    
    try: 
        for page_num , page in enumerate(doc):
            pages.append({
                "page_number":page_num+1,
                "content":page.get_text("text")
            })
    finally:
        doc.close()

    return pages

import re
import unicodedata

def clean_txt(text):
    
    if not text:
        return ""
    
    text = unicodedata.normalize("NFKC",text) # Unicode Normalization
    
    text = text.replace("\r\n", "\n").replace("\r", "\n") # Line Ending Normalization
    
    text = text.replace("\x00","") # NULL character removal
    
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text) # fixing word split due to hypheanation
    
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text) # Remove unnecessary spaces
    
    text = re.sub(r"[ \t]+", " ", text) # multiple spaces with a single space

    text = re.sub(r"\n{3,}", "\n\n", text) # Remove excessive blank lines
    
    text = text.strip() # Remove leading and trailing whitespace
    
    return text

    
    
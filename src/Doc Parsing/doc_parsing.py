import json
import datetime
from pathlib import Path

from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def parse_file(file_path, save_txt=True, save_json=True, out_dir="docs"):
    """
    Parse a document (PDF/DOCX/TXT) into text with metadata.
    
    Args:
        file_path (str or Path): Input file path
        save_txt (bool): Whether to save plain text file
        save_json (bool): Whether to save structured JSON
        out_dir (str): Output directory for results
    
    Returns:
        dict: structured output with metadata and text
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    all_text = []
    metadata = {
        "source_file": str(file_path),
        "file_name": file_path.name,
        "extension": ext,
        "parsed_time": datetime.datetime.now().isoformat(timespec="seconds"),
        "pages": []
    }

    try:
        if ext == ".pdf":
            try:
                text = pdf_extract_text(str(file_path))
                if text.strip():
                    # extract by page
                    for i, page_text in enumerate(text.split("\f")):
                        page_text = page_text.strip()
                        if page_text:
                            metadata["pages"].append({"page": i+1, "text": page_text})
                            all_text.append(page_text)
                else:
                    raise ValueError("Empty PDF text")
            except Exception:
                # fallback: OCR
                if OCR_AVAILABLE:
                    images = convert_from_path(str(file_path))
                    for i, img in enumerate(images):
                        page_text = pytesseract.image_to_string(img)
                        if page_text.strip():
                            metadata["pages"].append({"page": i+1, "text": page_text})
                            all_text.append(page_text)
                else:
                    raise RuntimeError("OCR not available but needed for scanned PDF.")

        elif ext == ".docx":
            doc = Document(str(file_path))
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    metadata["pages"].append({"page": i+1, "text": para.text})
                    all_text.append(para.text)

        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip():
                    metadata["pages"].append({"page": i+1, "text": line.strip()})
                    all_text.append(line.strip())
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        metadata["error"] = str(e)

    metadata["full_text"] = "\n\n".join(all_text)

    # save result
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base_name = file_path.stem

    if save_txt:
        txt_path = Path(out_dir) / f"{base_name}.txt"
        with open(txt_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(metadata["full_text"])

    if save_json:
        json_path = Path(out_dir) / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


# use case
if __name__ == "__main__":
    input_pdf = "data/Childrens Evolution of Care/State/Medicaid Updates/mu_no01_jan21_pr.pdf"
    result = parse_file(input_pdf, save_txt=True, save_json=True)
    print("Parsed file:", result["file_name"])
    print("Total chars:", len(result["full_text"]))
    print("Pages extracted:", len(result["pages"]))

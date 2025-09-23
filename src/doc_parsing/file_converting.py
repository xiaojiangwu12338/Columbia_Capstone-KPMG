from pathlib import Path
import win32com.client as win32

def convert_doc_to_docx(folder: Path, save_dir: Path) -> None:
    word = win32.Dispatch("Word.Application")
    word.Visible = False
    try:
        for doc_path in folder.rglob("*.doc"):
            # Skip temp files
            if doc_path.name.startswith("~$"):
                continue
            rel = doc_path.relative_to(folder)
            out_path = (save_dir / rel).with_suffix(".docx")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            doc = word.Documents.Open(str(doc_path))
            doc.SaveAs(str(out_path), FileFormat=16)  # .docx
            doc.Close()
    finally:
        word.Quit()

if __name__ == "__main__":
    Path(r"data\Childrens Evolution of Care\Federal\CMS Medicaid Manual_Converted").mkdir(parents=False,exist_ok=True)
    save_dir = Path(r"data\Childrens Evolution of Care\Federal\CMS Medicaid Manual_Converted").resolve()
    root = Path(r"data\Childrens Evolution of Care\Federal\CMS Medicaid Manual").resolve()
    for p in root.iterdir():
        res_dir = p.relative_to(root)
        (save_dir / res_dir).mkdir(parents=True,exist_ok=True)
        if p.is_dir():
            convert_doc_to_docx(p, save_dir / res_dir)
    #convert_doc_to_docx(root)
    print("Conversion complete")
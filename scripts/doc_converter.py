from doc_2_docx import convert_doc_to_docx
from pathlib import Path


if __name__ == "__main__":
    Path(r"data\raw\Childrens Evolution of Care\Federal\CMS Medicaid Manual_docx").mkdir(parents=False,exist_ok=True)
    save_dir = Path(r"data\raw\Childrens Evolution of Care\Federal\CMS Medicaid Manual_docx").resolve()
    root = Path(r"data\raw\Childrens Evolution of Care\Federal\CMS Medicaid Manual").resolve()
    for p in root.iterdir():
        res_dir = p.relative_to(root)
        (save_dir / res_dir).mkdir(parents=True,exist_ok=True)
        if p.is_dir():
            convert_doc_to_docx(p, save_dir / res_dir)
    #convert_doc_to_docx(root)
    print("Conversion complete")
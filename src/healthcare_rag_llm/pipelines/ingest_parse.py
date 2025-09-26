from pathlib import Path
from healthcare_rag_llm.doc_parsing import parse_file
from healthcare_rag_llm.utils.io import ensure_dir

def run_pipeline(raw_dir: str = "data/raw", out_dir: str = "data/processed",save_text=False,save_json=True):
    """
    Traverse documents in raw_dir:
    1. Process all files in the directory
    2. Call parse_file to extract structured data and save results
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    # fetch all files
    all_files = [f for f in raw_dir.rglob("*") if f.is_file()]
    
    if not all_files:
        print(f"[WARNING] No files found in {raw_dir}")
        return
    
    print(f"[INFO] Found {len(all_files)} files to process")
    
    for file in all_files:
        print(f"[INFO] Processing {file}")
        
        # check file format          
        try:
            # parse_file will automatically save JSON and TXT files to out_dir
            result = parse_file(file, save_txt=save_text, save_json=save_json, out_dir=str(out_dir))
            print(f"[OK] Processed: {file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to process {file}: {e}")
            continue


def run_cli():
    """
    Command-line entry point.
    After installation, run in terminal:
        ingest-parse
    """
    run_pipeline()

if __name__ == "__main__":
    run_pipeline()
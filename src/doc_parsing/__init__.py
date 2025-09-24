# src/doc_parsing/__init__.py
from .file_converting import convert_doc_to_docx  
from .doc_parsing import parse_file               

__all__ = ["convert_doc_to_docx", "parse_file"]
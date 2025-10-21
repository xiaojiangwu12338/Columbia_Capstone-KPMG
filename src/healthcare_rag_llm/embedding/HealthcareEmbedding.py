"""
Healthcare RAG Embedding Module

This module provides a simplified interface for BGE-M3 embeddings
specifically designed for healthcare document processing and retrieval.

Hardware Support:
    - NVIDIA GPUs: Automatic GPU acceleration (10-20x faster)
    - AMD/Intel GPUs: Automatic fallback to CPU
    - CPU only: Works fine, just slower

Compatibility:
    - Python: 3.9, 3.10, 3.11, 3.12
    - PyTorch: 2.2.0+
"""
import sys
import numpy as np
from typing import Optional
from FlagEmbedding import BGEM3FlagModel
import torch

class HealthcareEmbedding:
    """
    BGE-M3 embedding model with automatic hardware detection.

    This class automatically detects available hardware (GPU/CPU) and provides
    helpful feedback to users about performance and installation options.

    Args:
        use_fp16: Whether to use FP16 precision. If None, automatically determined:
                  - GPU available: True (faster, minimal accuracy loss)
                  - CPU only: False (FP32 for better numerical stability)
    """
    def __init__(self, use_fp16: Optional[bool] = None):
        # Check Python version
        py_version = sys.version_info
        if py_version < (3, 9):
            raise RuntimeError(
                f"Python 3.9+ is required, but you have Python {py_version.major}.{py_version.minor}.{py_version.micro}\n"
                f"Please upgrade Python or use a compatible virtual environment."
            )
        if py_version >= (3, 13):
            print(f"[WARNING] Python {py_version.major}.{py_version.minor} is very new")
            print(f"  Some dependencies may not be fully tested.")
            print(f"  Python 3.10-3.12 is recommended for best compatibility.\n")

        # Detect hardware and provide user feedback
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            print(f"[GPU] Detected: {gpu_name} ({gpu_memory:.1f} GB VRAM)")

            # Check if it's actually an NVIDIA GPU
            if "nvidia" not in gpu_name.lower():
                print(f"[WARNING] Non-NVIDIA GPU detected")
                print(f"  PyTorch CUDA support is optimized for NVIDIA GPUs")
                print(f"  Performance may be limited\n")

            # Auto-enable FP16 for GPU
            if use_fp16 is None:
                use_fp16 = True
            print(f"  Using FP16 precision for faster inference")
            print(f"  Expected performance: ~100-250 chunks/second\n")
        else:
            print(f"[CPU] No NVIDIA GPU detected, using CPU")
            print(f"  Performance: ~10-20x slower than GPU")
            print(f"  Expected performance: ~10-20 chunks/second")
            print(f"")
            print(f"  To enable GPU acceleration:")
            print(f"  1. Ensure you have an NVIDIA GPU (GTX 10 series or newer)")
            print(f"  2. Install CUDA-enabled PyTorch:")
            print(f"     pip uninstall torch torchvision torchaudio")
            print(f"     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print(f"")

            # Auto-disable FP16 for CPU
            if use_fp16 is None:
                use_fp16 = False

        # Load the model
        print(f"Loading BGE-M3 model (BAAI/bge-m3)...")
        print(f"  This may take 5-15 seconds on first run...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=use_fp16)
        print(f"[OK] Model loaded successfully!\n")
    
    def encode(self,text:list[str],return_dense=True,return_sparse=True,return_colbert_vecs=True):
        '''
        This function is used to get the embedding of the text using the BGE-M3 model.
        Since BGE-M3 can encode queries and documents and make them represented in the same semantic space
        so we can use the same function to encode both queries and documents.

        Args:
            text: list[str]
            return_dense: bool
            return_sparse: bool
            return_colbert_vecs: bool
        Returns:
            dictionary with keys: dense_vecs, sparse_vecs, colbert_vecs
            each value is a list[np.ndarray]
        '''
        return self.model.encode(text,return_dense=return_dense,return_sparse=return_sparse,return_colbert_vecs=return_colbert_vecs)

if __name__ == "__main__":
    documents = ["C-YES conducts the HCBS/Level of Care Eligibility Determination for children opting out of Health Home."]
    queries = ["Providers seeking prior authorization can contact the Magellan Clinical Call Center."]
    embedding = HealthcareEmbedding()
    print(embedding.encode(documents)['dense_vecs'][0]@embedding.encode(queries)['dense_vecs'][0].T)

    documents = ["Hospitals must ensure the accuracy of patient discharge status coding on Medicaid claims."]
    queries = ["Hospitals must correctly code whether patients are transferred or discharged, since this affects Medicaid payments."]
    print(embedding.encode(documents)['dense_vecs'][0]@embedding.encode(queries)['dense_vecs'][0].T)
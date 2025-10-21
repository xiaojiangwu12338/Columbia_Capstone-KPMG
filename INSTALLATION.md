# Installation Guide

Complete installation instructions for the Healthcare RAG LLM system.

## System Requirements

### Python Version
- **Required**: Python 3.9, 3.10, 3.11, or 3.12
- **Not supported**: Python 3.13+ (PyTorch compatibility issues)
- **Recommended**: Python 3.11 (best balance of features and stability)

Check your Python version:
```bash
python --version
```

### Hardware Requirements

#### GPU (Optional, but Highly Recommended)

**Supported GPUs:**
- ✅ **NVIDIA GPUs** (GTX 10 series and newer):
  - RTX 40 series (4090, 4080, 4070...)
  - RTX 30 series (3090, 3080, 3070...)
  - RTX 20 series (2080 Ti, 2070, 2060...)
  - GTX 16 series (1660, 1650...)
  - GTX 10 series (1080 Ti, 1070, 1060...)

**Performance Comparison:**
| Hardware | Embedding Speed | 5000 chunks processing time |
|----------|----------------|---------------------------|
| RTX 4090 | ~250 chunks/sec | ~20 seconds |
| RTX 3080 | ~200 chunks/sec | ~25 seconds |
| RTX 2070 | ~150 chunks/sec | ~35 seconds |
| GTX 1660 | ~100 chunks/sec | ~50 seconds |
| **CPU only** | **~10-20 chunks/sec** | **250-500 seconds** |

**Unsupported (will fallback to CPU):**
- ❌ AMD GPUs (Radeon RX series) - requires ROCm, not covered here
- ❌ Intel Arc GPUs - limited PyTorch support
- ❌ Integrated graphics (Intel UHD, AMD Vega)

#### CPU & Memory
- **CPU**: Modern multi-core processor (4+ cores recommended)
- **RAM**: Minimum 8GB, 16GB+ recommended for large document processing
- **Storage**: 10GB free space (for models and data)

## Installation Steps

### Step 1: Clone the Repository

```bash
git clone git@github.com:xiaojiangwu12338/Columbia_Capstone-KPMG.git
cd Columbia_Capstone-KPMG
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install PyTorch

**This is the most important step!** Choose the right PyTorch version for your system.

#### Option A: Universal Installation (Recommended)

**This works on ALL systems** - automatically uses GPU if available, falls back to CPU otherwise:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Why cu121?**
- `cu121` = CUDA 12.1
- Compatible with NVIDIA drivers CUDA 11.8 and newer
- Works on systems **with or without** NVIDIA GPUs
- If no GPU is present, it automatically uses CPU

#### Option B: CPU-Only Installation (Smaller Download)

If you're **certain** you don't have an NVIDIA GPU or want a smaller download:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Warning**: This version will **never** use GPU, even if you add one later.

### Step 4: Install Other Dependencies

```bash
pip install -e .
```

This installs all required packages from `pyproject.toml`.

### Step 4.5: Download NLTK Data (Required for Semantic Chunking)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

This downloads sentence tokenizers needed for semantic chunking.

### Step 5: Install System Dependencies

#### Windows
1. **Tesseract OCR**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. **Microsoft Word** (for .doc conversion) OR **LibreOffice**

#### macOS
```bash
brew install tesseract libreoffice
```

#### Linux
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr libreoffice
```

### Step 6: Set Up Neo4j Database

```bash
cd docker
cp .env.example .env
# Edit .env and set your Neo4j password
docker compose up -d
```

Verify Neo4j is running: http://localhost:7474

### Step 7: Verify Installation

Run this command to check your hardware setup:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Device: {\"GPU - \" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

**Expected output with GPU:**
```
PyTorch version: 2.2.2+cu121
Device: GPU - NVIDIA GeForce RTX 2070
```

**Expected output without GPU:**
```
PyTorch version: 2.2.2+cu121
Device: CPU
```

Both are correct! The system will automatically use whatever hardware you have.

## Troubleshooting

### "No NVIDIA GPU detected" but you have an NVIDIA GPU

**Cause**: You installed the CPU-only version of PyTorch.

**Solution**:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA out of memory" error

**Cause**: Your GPU doesn't have enough VRAM for the current batch size.

**Solution**: Reduce the batch size in your scripts:
```python
# In ingest_graph.py or evaluation scripts
ingest_chunks(..., embedding_batch_size=16)  # Reduce from 32
```

Or use smaller models (not recommended, reduces accuracy):
```python
embedder = HealthcareEmbedding(use_fp16=True)  # FP16 uses less memory
```

### "Python 3.13 is very new" warning

**Cause**: You're using Python 3.13, which is too new.

**Solution**: Downgrade to Python 3.11 or 3.12:
```bash
# Create new environment with specific Python version
conda create -n healthcare_rag python=3.11
conda activate healthcare_rag
# Then repeat installation steps
```

### Very slow embedding generation (< 10 chunks/sec on GPU)

**Possible causes:**
1. Using CPU version of PyTorch (see first troubleshooting item)
2. GPU is being used by another process
3. Batch size is too small

**Check GPU utilization:**
```bash
nvidia-smi
```

Should show high GPU utilization when running embedding generation.

### "ImportError: No module named 'pywin32'" (Windows only)

**Solution**:
```bash
pip install pywin32
```

This is only needed for .doc to .docx conversion on Windows.

## Performance Optimization Tips

### For GPU Users

1. **Increase batch size** (if you have enough VRAM):
   ```python
   embedding_batch_size=64  # or even 128 for high-end GPUs
   ```

2. **Close other GPU-intensive applications** (games, video editing, etc.)

3. **Monitor GPU temperature** - thermal throttling can reduce performance

### For CPU Users

1. **Close unnecessary applications** to free up RAM
2. **Use smaller batch sizes** to avoid memory issues:
   ```python
   embedding_batch_size=16
   ```
3. **Be patient** - CPU processing is 10-20x slower but will complete

## Next Steps

After successful installation:

1. **Test the system**:
   ```bash
   python scripts/test_neo4j.py
   ```

2. **Run a simple ingestion**:
   ```bash
   python scripts/ingest_graph.py --chunk_dir data/chunks/asterisk_chunking_result
   ```

3. **Start the web interface**:
   ```bash
   streamlit run frontend/app.py
   ```

See [README.md](README.md) for detailed usage instructions.

## Getting Help

- **Documentation**: See README.md and CLAUDE.md
- **Issues**: Report bugs on GitHub Issues
- **Common errors**: Check this troubleshooting section first

## Hardware Recommendations

### For Development (Frequent Testing)
- **Minimum**: NVIDIA GTX 1660 or better
- **Recommended**: NVIDIA RTX 3060 or better
- **Ideal**: NVIDIA RTX 4070 or better

### For Production/Deployment
- **Server**: NVIDIA Tesla T4, V100, or A100
- **Workstation**: NVIDIA RTX 4090 or RTX 6000
- **Cloud**: AWS p3.2xlarge, Azure NC6s v3, or Google Cloud N1 with T4

### For CPU-Only Users
Still usable, but expect longer processing times. Consider:
- Using smaller document sets for testing
- Running overnight for large batches
- Cloud GPU instances for large-scale processing (e.g., Google Colab Pro)

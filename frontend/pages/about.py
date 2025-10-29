import streamlit as st

st.set_page_config(page_title="About – NYS Policies Assistant")

st.title("📘 About the Project")
st.markdown("""
**Columbia University x KPMG**  
*Intelligent Document Analysis for Healthcare Programs Using LLMs and RAG*

---

### 🎯 Project Overview
Government healthcare programs, such as New York State Medicaid, involve complex eligibility criteria 
spread across multiple interrelated policy documents.  
Our project leverages **Large Language Models (LLMs)** and **Retrieval-Augmented Generation (RAG)** 
to simplify policy navigation and question answering.

Users can ask questions such as:
> “When did redetermination begin for the COVID-19 Public Health Emergency unwind in New York State?”

and receive grounded, cited answers from authoritative Medicaid documents.

---
            
### 🔗 GitHub Repository
[![GitHub](https://img.shields.io/badge/GitHub-Columbia__Capstone--KPMG-blue?logo=github)](https://github.com/xiaojiangwu12338/Columbia_Capstone-KPMG)

Full project code and documentation can be found on GitHub:  
👉 [https://github.com/xiaojiangwu12338/Columbia_Capstone-KPMG](https://github.com/xiaojiangwu12338/Columbia_Capstone-KPMG)

---
            
### 👥 Team Members
- **Ruoheng Du**
- **Xiaojiang Wu**
- **Yijian Liu**
- **Shashwat Kumar**

### 🧑‍🏫 Mentors
- **Jim Leach**, KPMG  
- **John Trombley**, KPMG

---

© 2025 Columbia DSI x KPMG | For educational use only.
""")

# ### ⚙️ Technologies
# - **LLM & RAG Pipeline:** Retrieval, Reranking, Context Construction  
# - **Knowledge Graph:** Neo4j  
# - **Embedding:** BGE-M3  
# - **Frontend:** Streamlit UI  
# - **Backend Integration:** Python API (LLMClient, ResponseGenerator)

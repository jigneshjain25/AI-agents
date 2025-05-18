# setup


##  Dependencies
```
pip install sentence-transformers chromadb pymupdf python-docx pandas requests pytesseract pillow google-generativeai
sudo apt install tesseract-ocr
```

## (not used) Ollama
To run llm models locally.

```
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Install models
ollama pull nomic-embed-text
ollama pull llama2
ollama serve
```

# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Kaggle dataset automatically
python download_dataset.py

# 3. Set DATASET_PATH in .env (script tells you the path)
# 4. Run FastAPI
# python -m uvicorn main:app --reload --port 8000
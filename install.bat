@echo off
echo Checking Python version...
python --version

echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel

echo Installing dependencies...
pip install -e .

echo Downloading NLTK data...
python -c "import nltk; nltk.download('vader_lexicon')"

echo Downloading transformer models...
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')"

echo Installation complete!
echo To activate the environment, run: venv\Scripts\activate.bat
echo To start the server, run: python app.py 
"""
Test script to verify all dependencies are properly installed
Run this after setup to ensure everything works correctly
"""

import sys

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        # Core libraries
        import pandas as pd
        print("✅ pandas")
        
        import numpy as np
        print("✅ numpy")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
        
        import seaborn as sns
        print("✅ seaborn")
        
        # Machine Learning
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        print("✅ scikit-learn")
        
        # NLP
        import spacy
        print("✅ spacy")
        
        from wordcloud import WordCloud
        print("✅ wordcloud")
        
        # Deep Learning
        import torch
        print(f"✅ torch (version: {torch.__version__})")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        print("✅ transformers")
        
        from datasets import load_dataset
        print("✅ datasets")
        
        import accelerate
        print("✅ accelerate")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False


def test_spacy_model():
    """Test if SpaCy French model is installed"""
    print("\n🧪 Testing SpaCy French model...")
    
    try:
        import spacy
        nlp = spacy.load("fr_core_news_sm")
        doc = nlp("Bonjour, ceci est un test.")
        print("✅ SpaCy French model loaded successfully")
        return True
    except OSError:
        print("❌ SpaCy French model not found")
        print("   Run: python -m spacy download fr_core_news_sm")
        return False


def test_jupyter_kernel():
    """Test if Jupyter kernel is installed"""
    print("\n🧪 Testing Jupyter kernel...")
    
    try:
        import jupyter_client
        km = jupyter_client.kernelspec.KernelSpecManager()
        kernels = km.find_kernel_specs()
        
        if 'eda_venv' in kernels:
            print("✅ Jupyter kernel 'eda_venv' found")
            return True
        else:
            print("⚠️  Jupyter kernel 'eda_venv' not found")
            print("   Run: python -m ipykernel install --user --name=eda_venv")
            return False
    except Exception as e:
        print(f"❌ Error checking kernel: {e}")
        return False


def main():
    print("=" * 60)
    print("🔍 Sentiment Analysis Project - Dependency Check")
    print("=" * 60)
    print()
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test SpaCy model
    results.append(test_spacy_model())
    
    # Test Jupyter kernel
    results.append(test_jupyter_kernel())
    
    # Summary
    print("\n" + "=" * 60)
    if all(results):
        print("✅ All tests passed! You're ready to go!")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook")
        print("  2. Open: sentiment_analysis.ipynb")
        print("  3. Select kernel: Python (EDA)")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()

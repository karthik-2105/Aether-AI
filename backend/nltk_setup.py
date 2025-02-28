import nltk
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class NLTKDownloader:
    REQUIRED_PACKAGES = [
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet',
        'stopwords'
    ]
    
    @staticmethod
    def check_nltk_data():
        """Check if NLTK data is already downloaded"""
        nltk_data_dir = Path(nltk.data.find('.')).parent
        
        for package in NLTKDownloader.REQUIRED_PACKAGES:
            if not os.path.exists(nltk_data_dir / package):
                return False
        return True
    
    @staticmethod
    def download_nltk_data():
        """Download required NLTK data if not present"""
        try:
            if not NLTKDownloader.check_nltk_data():
                logger.info("Downloading required NLTK data...")
                for package in NLTKDownloader.REQUIRED_PACKAGES:
                    nltk.download(package, quiet=True)
                logger.info("NLTK data downloaded successfully")
            else:
                logger.info("NLTK data already present")
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {str(e)}")
            raise
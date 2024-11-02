import os
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Obtener la ruta absoluta al directorio actual
basedir = os.path.abspath(os.path.dirname(__file__))

# Cargar .env desde el directorio correcto
dotenv_path = os.path.join(basedir, '.env')
load_dotenv(dotenv_path)

# Log para debug
logger.info(f"Loading .env from: {dotenv_path}")
logger.info(f"Environment variables loaded: {list(os.environ.keys())}")

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
    
    def __init__(self):
        # Verificar que tenemos la API key
        if not self.PERPLEXITY_API_KEY:
            logger.error("PERPLEXITY_API_KEY no encontrada en las variables de entorno")
        else:
            # Log seguro (solo primeros caracteres)
            key_preview = f"{self.PERPLEXITY_API_KEY[:4]}..." if len(self.PERPLEXITY_API_KEY) > 4 else "***"
            logger.info(f"PERPLEXITY_API_KEY loaded: {key_preview}")

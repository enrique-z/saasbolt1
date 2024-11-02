from flask import Flask, render_template, redirect, url_for
from config import Config
from modules.keyword_research import keyword_research
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config.from_object(Config)

# Add secret key for session
app.secret_key = Config.SECRET_KEY

# Register blueprint
app.register_blueprint(keyword_research)

# Add this debug route temporarily
@app.route('/debug')
def debug():
    static_dir = os.path.join(app.root_path, 'static')
    images_dir = os.path.join(static_dir, 'images')
    files = {}
    if os.path.exists(images_dir):
        for file in os.listdir(images_dir):
            file_path = os.path.join(images_dir, file)
            files[file] = {
                'size': os.path.getsize(file_path),
                'path': file_path
            }
    return {
        'static_path': static_dir,
        'images_path': images_dir,
        'files': files
    }

@app.route('/')
def index():
    return render_template('index.html')

# Add a redirect route
@app.route('/redirect-to-keyword')
def redirect_to_keyword():
    return redirect(url_for('keyword_research.keyword_research_view'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from member5_evaluation.inference_pipeline import run_inference
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs/inference'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Important checkpoints path
SIAMESE_CKPT = "checkpoints/siamese_best.pth"
CLASSIFIER_CKPT = "checkpoints/classifier_best.pth"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pre_img' not in request.files or 'post_img' not in request.files:
            return "Please upload BOTH 'Before' and 'After' images", 400
        
        pre_file = request.files['pre_img']
        post_file = request.files['post_img']
        
        if pre_file.filename == '' or post_file.filename == '':
            return "No selected file", 400
            
        pre_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pre_file.filename))
        post_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(post_file.filename))
        
        pre_file.save(pre_path)
        post_file.save(post_path)

        # Ensure checkpoints exist (Mock if missing for demonstration)
        # However, run_inference will fail if they don't exist.
        if not os.path.exists(SIAMESE_CKPT) or not os.path.exists(CLASSIFIER_CKPT):
            return "Error: Model checkpoints not found. Please ensure training is complete for BOTH Siamese and Classifier.", 500

        try:
            timestamp_dir = os.path.join(app.config['OUTPUT_FOLDER'], str(int(time.time())))
            os.makedirs(timestamp_dir, exist_ok=True)
            
            # Run inference
            result = run_inference(
                pre_img_path=pre_path,
                post_img_path=post_path,
                siamese_ckpt=SIAMESE_CKPT,
                classifier_ckpt=CLASSIFIER_CKPT,
                output_dir=timestamp_dir,
                device="cuda",
            )
            
            map_name = "damage_map.html"
            folder_name = os.path.basename(timestamp_dir)
            
            return redirect(url_for('show_result', folder=folder_name, map_file=map_name))
        except Exception as e:
            return f"Error during inference: {str(e)}", 500

    return render_template('index.html')

@app.route('/result/<folder>/<map_file>')
def show_result(folder, map_file):
    # Serve the interactive folium HTML map inside an iframe, or directly
    return render_template('result.html', folder=folder, map_file=map_file)

@app.route('/map/<folder>/<map_file>')
def serve_map(folder, map_file):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], folder), map_file)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

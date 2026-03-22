import os
import json
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from member5_evaluation.inference_pipeline import run_inference
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs/inference'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB max upload

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

        pre_file  = request.files['pre_img']
        post_file = request.files['post_img']

        if pre_file.filename == '' or post_file.filename == '':
            return "No selected file", 400

        pre_filename  = secure_filename(pre_file.filename)
        post_filename = secure_filename(post_file.filename)
        pre_path  = os.path.join(app.config['UPLOAD_FOLDER'], pre_filename)
        post_path = os.path.join(app.config['UPLOAD_FOLDER'], post_filename)

        pre_file.save(pre_path)
        post_file.save(post_path)

        if not os.path.exists(SIAMESE_CKPT) or not os.path.exists(CLASSIFIER_CKPT):
            return "Error: Model checkpoints not found.", 500

        try:
            timestamp_dir = os.path.join(app.config['OUTPUT_FOLDER'], str(int(time.time())))
            os.makedirs(timestamp_dir, exist_ok=True)

            result = run_inference(
                pre_img_path=pre_path,
                post_img_path=post_path,
                siamese_ckpt=SIAMESE_CKPT,
                classifier_ckpt=CLASSIFIER_CKPT,
                output_dir=timestamp_dir,
                device="cuda",
                map_threshold=0.35,
            )

            folder_name = os.path.basename(timestamp_dir)

            # Read damage counts from GeoJSON properties
            gj_path = result.get('geojson_path', '')
            stats = {'no_damage': 0, 'minor_damage': 0, 'major_damage': 0, 'destroyed': 0}
            total = 0
            if os.path.exists(gj_path):
                with open(gj_path) as f:
                    gj = json.load(f)
                props = gj.get('properties', {})
                stats['no_damage']    = props.get('no_damage', 0)
                stats['minor_damage'] = props.get('minor_damage', 0)
                stats['major_damage'] = props.get('major_damage', 0)
                stats['destroyed']    = props.get('destroyed', 0)
                total = sum(stats.values())

            return redirect(url_for(
                'show_result',
                folder=folder_name,
                map_file='damage_map.html',
                pre=pre_filename,
                post=post_filename,
                no_dmg=stats['no_damage'],
                minor=stats['minor_damage'],
                major=stats['major_damage'],
                destroyed=stats['destroyed'],
                total=total,
            ))
        except Exception as e:
            import traceback
            return f"<pre>Error during inference:\n{traceback.format_exc()}</pre>", 500

    return render_template('index.html')


@app.route('/result/<folder>/<map_file>')
def show_result(folder, map_file):
    return render_template(
        'result.html',
        folder=folder,
        map_file=map_file,
        pre=request.args.get('pre', ''),
        post=request.args.get('post', ''),
        no_dmg=int(request.args.get('no_dmg', 0)),
        minor=int(request.args.get('minor', 0)),
        major=int(request.args.get('major', 0)),
        destroyed=int(request.args.get('destroyed', 0)),
        total=int(request.args.get('total', 0)),
    )


@app.route('/map/<folder>/<map_file>')
def serve_map(folder, map_file):
    return send_from_directory(os.path.join(app.config['OUTPUT_FOLDER'], folder), map_file)


@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)

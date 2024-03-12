from flask import Flask, request, render_template, redirect, url_for, send_file
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
GENERATED_IMAGES_FOLDER = 'static/generated_images'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_IMAGES_FOLDER'] = GENERATED_IMAGES_FOLDER

def get_next_uploaded_image_name(folder):
    count = 1
    while True:
        image_name = f"img_{count:03d}.jpg"
        if not os.path.exists(os.path.join(folder, image_name)):
            return image_name
        count += 1

def get_next_generated_image_name(folder):
    count = 1
    while True:
        image_name = f"gen_img_{count:03d}.jpg"
        if not os.path.exists(os.path.join(folder, image_name)):
            return image_name
        count += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def handle_form_submission():
    # Extract form data
    style_description = request.form['styleDescription']
    strength = float(request.form['strength'])
    guidance_scale = float(request.form['guidanceScale'])
    num_inference_steps = int(request.form['numInferenceSteps'])

    # Handle file upload
    uploaded_file = request.files['imageUpload']

    # Save the uploaded file with a different naming convention
    uploaded_image_name = get_next_uploaded_image_name(app.config['UPLOAD_FOLDER'])
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_name))

    # Read the image data
    uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image_name)
    init_image = Image.open(uploaded_image_path).convert("RGB")

    # Set the prompt
    prompt = "give the image a taj style base but keep the core elements intact, just apply the style with originality preserved, give it a higher resolution render- enhance it"

    # Generate the image
    model_id_or_path = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    images = pipe(prompt=style_description, image=init_image, strength=strength, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, scheduler='K-EULER').images

    # Save the generated image
    generated_image_name = get_next_generated_image_name(app.config['GENERATED_IMAGES_FOLDER'])
    generated_image_path = os.path.join(app.config['GENERATED_IMAGES_FOLDER'], generated_image_name)
    images[0].save(generated_image_path)

    # Redirect to the result page
    return redirect(url_for('result', uploaded_image_name=uploaded_image_name, generated_image_name=generated_image_name))
    
@app.route('/result')
def result():
    uploaded_image_name = request.args.get('uploaded_image_name', '')
    generated_image_name = request.args.get('generated_image_name', '')
    uploaded_image_path = uploaded_image_path = url_for('handle_form_submission', filename=uploaded_image_name)

    generated_image_path = url_for('static', filename=f'generated_images/{generated_image_name}')
    return render_template('result.html', uploaded_image_path=uploaded_image_path, generated_image_path=generated_image_path)

if __name__ == '__main__':
    # Ensure the 'uploads' and 'executed_notebooks' directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('executed_notebooks', exist_ok=True)

    app.run(debug=False)

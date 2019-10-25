from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf
from skimage.io import imsave
from skimage.transform import resize
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from keras.applications.inception_resnet_v2 import preprocess_input
from PIL import Image,ImageChops
import logging 


global graph
graph = tf.get_default_graph()
app = Flask(__name__)
app.secret_key = "hello"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
model = load_model('trained-model.h5')
UPLOAD_FOLDER = '/home/nubaf/Git-Projects/colorization/files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
files = [f for f in os.listdir('.') if os.path.isfile(f)]
checkInception = False
for f in files:
    if f == "inception.h5":
        checkInception = True
        inception = load_model('inception.h5', compile=False)
        break
if not checkInception:
    inception = InceptionResNetV2(weights='imagenet', include_top=True)
    inception.save('inception.h5')
inception.graph = graph


def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
        return embed


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        try:
            url = request.form['url']
            if 'examples' in url:
                color_file = process(url)
                return render_template('index.html', res='static/examples/girl.jpg')
        # check if the post request has the file part
        except:
            logging.exception('')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            color_file = process(file.filename)
            return render_template('index.html', og=color_file[0], res=color_file[1])


    return render_template('index.html')

def process(img):
    if 'examples' in img:
        im = Image.open(img)
        name = img.split('.')[0].split('/')[-1]
    else:
        im = Image.open('files/' + img)
        name = img.split('.')[0]
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(256)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (256, 256))
    new_im.paste(im, ((256-new_size[0])//2,(256-new_size[1])//2))
    new_im.save('static/processed_png/' + name + ".png","PNG")
    a = np.array(img_to_array(load_img('static/processed_png/' + name +'.png')))
    a = a.reshape(1,256,256,3)
    #gray_me = gray2rgb(rgb2gray(1.0/255*a))
    color_me_embed = create_inception_embedding(a)
    a = rgb2lab(1.0/255*a)[:,:,:,0]
    a = a.reshape(a.shape+(1,))
    with graph.as_default():
        output = model.predict([a, color_me_embed])
        output = output * 128
        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:,:,0] = a[i][:,:,0]
            cur[:,:,1:] = output[i]
            imsave(f'static/colored_img/{name}.png',(lab2rgb(cur)))
            trim(Image.open(f'static/processed_png/{name}.png')).save(f'static/processed_png/{name}.png')
            trim(Image.open(f'static/colored_img/{name}.png')).save(f'static/colored_img/{name}.png')
            return (f'static/processed_png/{name}.png',f'static/colored_img/{name}.png') 


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

if __name__ == "__main__":
    app.run(debug=True)

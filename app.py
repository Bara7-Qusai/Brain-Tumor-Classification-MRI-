from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import numpy as np
import cv2
from PIL import Image
import io

app = Flask(__name__)

# تحميل النموذج
model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model = tf.keras.layers.GlobalAveragePooling2D()(model.output)
model = tf.keras.layers.BatchNormalization()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4, activation='softmax')(model)
model = tf.keras.models.Model(inputs=model.input, outputs=model)
model.load_weights('resnet_bn.h5')

labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

@app.route('/', methods=['GET', 'POST'])
def index():
    label = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file.stream)
            img = img.resize((150, 150))
            img = np.array(img)
            if img.shape[2] == 4:  # convert RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = img / 255.0
            img = img.reshape(1, 150, 150, 3)
            pred = model.predict(img)
            pred = np.argmax(pred, axis=1)[0]
            label = labels[pred]
    return render_template('index.html', label=label)

if __name__ == '__main__':
    app.run(debug=True)

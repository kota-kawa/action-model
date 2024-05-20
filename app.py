from flask import Flask, request, jsonify, render_template, redirect
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from agent import task_agent
import langchain_pdf

app = Flask(__name__)


user_credentials = {
'user1': 'Z1MowIuz1e4I',
'user2': 'sKWEHnM9gOP6',
'user3': 'tWtYtq6zGoSN',
}


# Load the trained model
model = load_model('face_model.h5')

# Get the names of the labels
dataset_path="faces_dataset"
label_names = [person_folder for person_folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, person_folder))]

redirect_flag = False
person_name = None 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        global redirect_flag
        global person_name  
        global password
        redirect_flag = False
        person_name = None 
        password = None

        # Get the image from the POST request
        image_b64 = request.json['image'].split(',')[1]
        image_data = base64.b64decode(image_b64)
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Process the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            face_img = face_img.astype("float") / 255.0
            face_img = img_to_array(face_img)
            face_img = np.expand_dims(face_img, axis=0)

            preds = model.predict(face_img)[0]
            label = np.argmax(preds)
            person_name = label_names[int(label)]
            proba = preds[label]

            label = "{}: {:.2f}%".format(person_name, proba * 100)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if int(proba * 100) > 60:
                redirect_flag = True

        # Convert the processed image to base64
        _, buffer = cv2.imencode('.jpg', image)
        image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Return the processed image
        return jsonify({'image': 'data:image/jpg;base64,' + image_b64})

    return render_template('camera_login.html')

@app.route('/check_face')
def check_face():
    global redirect_flag
    global person_name
    if redirect_flag:
        redirect_flag = False
        password = user_credentials.get(person_name, None)  # Extract password from dictionary using label name as keyword
        return jsonify({'redirect': True, 'id': person_name, 'password': password})
    else:
        return jsonify({'redirect': False})

@app.route('/login_check', methods=['GET', 'POST'])
def login_check():
    # Get id and password from form
    id = request.form['id']
    password = request.form['password']
    if id in user_credentials and user_credentials[id] == password:
        return redirect("/chat")
    else:
        return "incorrect"
    
@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/post', methods=['POST'])
def post():
    data_dic= {
    "color": "",
    "change_object1": "",
    "change_object2": "",
    "play_animation":"",
    "form_text":"",
    "new_object":"",
    "new_object_color":"",
    "delete_object":"",
    "bold_text":"",
    "word_file":"",
    }  
    message = request.form['message']
    pdf_file = request.files['pdf_file']
     # Check if file was uploaded
    if pdf_file:
        pdf_file = request.files['pdf_file']
        response = langchain_pdf.chain_main(message, pdf_file)
    else:
        response,data_dic = task_agent.main_agent(message)

    return jsonify({'response': response, 'data_dic': data_dic}) 

app.register_blueprint(task_agent.app)

if __name__ == '__main__':
    app.run(debug=True)
   

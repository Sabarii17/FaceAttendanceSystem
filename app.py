from flask import Flask, render_template, request
import face_recognition
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STUDENT_FOLDER = 'students'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

known_face_encodings = []
known_face_names = []

# Load student images and encode
for filename in os.listdir(STUDENT_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        student_path = os.path.join(STUDENT_FOLDER, filename)
        image_bgr = cv2.imread(student_path)
        if image_bgr is None:
            print(f"❌ Failed to load {filename}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(image_rgb)
        print(f"Encoding for {filename}: {len(encodings)} face(s) found")
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

CONFIDENCE_THRESHOLD = 0.6

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'class_photos[]' not in request.files:
        return "No file part"
    files = request.files.getlist('class_photos[]')
    if not files or files[0].filename == '':
        return "No selected files"

    present_students = set()

    for file in files:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        image_bgr = cv2.imread(filepath)
        if image_bgr is None:
            print(f"❌ Failed to load uploaded photo: {file.filename}")
            continue

        classroom_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(classroom_image)
        face_encodings = face_recognition.face_encodings(classroom_image, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < CONFIDENCE_THRESHOLD:
                name = known_face_names[best_match_index]
                present_students.add(name)
            else:
                print(f"❌ No valid match for face, or confidence too low: {face_distances[best_match_index]}")

    absent_students = set(known_face_names) - present_students
    total_students = len(known_face_names)
    num_present = len(present_students)
    num_absent = len(absent_students)

    return render_template(
        'result.html',
        present=sorted(present_students),
        absent=sorted(absent_students),
        total_students=total_students,
        num_present=num_present,
        num_absent=num_absent
    )

if __name__ == '__main__':
    app.run(debug=True)

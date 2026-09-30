import os
import csv
from datetime import datetime

import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Face Attendance", layout="centered")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(ellipse at top, #1b2735 0%, #090a0f 100%);
    overflow: hidden;
}

.stars, .stars2, .stars3 {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    z-index: 0;
    background-repeat: repeat;
}

.stars {
    background-image:
        radial-gradient(1.5px 1.5px at 20px 30px, white, transparent),
        radial-gradient(1px 1px at 90px 120px, white, transparent),
        radial-gradient(1.5px 1.5px at 150px 60px, white, transparent),
        radial-gradient(1px 1px at 210px 180px, white, transparent),
        radial-gradient(1.5px 1.5px at 280px 40px, white, transparent);
    background-size: 300px 200px;
    animation: drift 80s linear infinite, twinkle 3s ease-in-out infinite alternate;
}

.stars2 {
    background-image:
        radial-gradient(1px 1px at 50px 90px, #a0d8ff, transparent),
        radial-gradient(1.5px 1.5px at 130px 30px, #a0d8ff, transparent),
        radial-gradient(1px 1px at 200px 140px, #a0d8ff, transparent);
    background-size: 250px 250px;
    animation: drift 120s linear infinite, twinkle 4s ease-in-out infinite alternate-reverse;
    opacity: 0.7;
}

.stars3 {
    background-image:
        radial-gradient(2px 2px at 100px 50px, white, transparent),
        radial-gradient(2px 2px at 250px 150px, white, transparent);
    background-size: 400px 400px;
    animation: drift 200s linear infinite;
    opacity: 0.9;
}

@keyframes drift {
    from { transform: translateY(0px); }
    to { transform: translateY(-1000px); }
}

@keyframes twinkle {
    from { opacity: 0.4; }
    to { opacity: 1; }
}

.shooting-star {
    position: fixed;
    top: 10%;
    left: 80%;
    width: 2px;
    height: 2px;
    background: white;
    border-radius: 50%;
    box-shadow: 0 0 6px 2px white;
    animation: shoot 6s linear infinite;
    z-index: 0;
}

.shooting-star::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 120px;
    height: 1px;
    background: linear-gradient(to left, white, transparent);
    transform: translateY(-50%);
}

@keyframes shoot {
    0% { transform: translate(0, 0); opacity: 1; }
    15% { transform: translate(-400px, 200px); opacity: 0; }
    100% { transform: translate(-400px, 200px); opacity: 0; }
}

h1, h2, h3, p, label, .stMarkdown {
    color: white !important;
    position: relative;
    z-index: 1;
}
</style>

<div class="stars"></div>
<div class="stars2"></div>
<div class="stars3"></div>
<div class="shooting-star"></div>
""", unsafe_allow_html=True)

IMAGES_FOLDER = "ImagesAttendance"
ATTENDANCE_FILE = "Attendance.csv"
MATCH_THRESHOLD = 0.5


def load_known_faces(folder):
    encodings, names = [], []
    for file_name in os.listdir(folder):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = Image.open(os.path.join(folder, file_name)).convert("RGB")
        image = np.ascontiguousarray(np.array(image, dtype=np.uint8))
        found = face_recognition.face_encodings(image)
        if found:
            encodings.append(found[0])
            names.append(file_name.split(".")[0].strip())
    return encodings, names


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Name", "Date", "Time"])

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        already_marked = any(
            row[0] == name and row[1] == today for row in csv.reader(f) if row
        )

    if not already_marked:
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            csv.writer(f).writerow([name, today, time_now])
        return True
    return False


st.title("Face Recognition Attendance System")
menu = st.sidebar.radio("Menu", ["Take Attendance", "View Records", "Register New Person"])

if menu == "Take Attendance":
    st.header("Take Attendance")

    if "camera_on" not in st.session_state:
        st.session_state.camera_on = False

    if not st.session_state.camera_on:
        if st.button("Start Camera"):
            st.session_state.camera_on = True
            st.rerun()
        photo = None
    else:
        if st.button("Stop Camera"):
            st.session_state.camera_on = False
            st.rerun()
        photo = st.camera_input("Look at the camera and take a photo")

    if photo is not None:
        known_encodings, known_names = load_known_faces(IMAGES_FOLDER)

        image = Image.open(photo).convert("RGB")
        image = np.ascontiguousarray(np.array(image, dtype=np.uint8))
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            st.error("No face detected. Try again with better lighting.")
        else:
            distances = face_recognition.face_distance(known_encodings, face_encodings[0])
            best = int(np.argmin(distances))

            if distances[best] < MATCH_THRESHOLD:
                name = known_names[best]
                newly_marked = mark_attendance(name)
                if newly_marked:
                    st.success(f"Attendance marked for {name}!")
                else:
                    st.info(f"{name}, your attendance was already marked today.")
            else:
                st.error("Face not recognized.")

elif menu == "View Records":
    st.header("Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df)
    else:
        st.write("No attendance records yet.")

elif menu == "Register New Person":
    st.header("Register New Person")

    name = st.text_input("Enter the person's name")
    photo = st.camera_input("Take a clear photo of their face")

    if st.button("Save this person"):
        if not name.strip():
            st.error("Please type a name first.")
        elif photo is None:
            st.error("Please take a photo first.")
        else:
            image = Image.open(photo).convert("RGB")
            check_array = np.ascontiguousarray(np.array(image, dtype=np.uint8))
            found = face_recognition.face_encodings(check_array)

            if not found:
                st.error("No face detected in that photo. Try again with better lighting.")
            else:
                save_path = os.path.join(IMAGES_FOLDER, f"{name.strip()}.jpg")
                image.save(save_path)
                st.success(f"{name.strip()} has been registered! They can now use Take Attendance.")
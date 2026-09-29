import os
import csv
from datetime import datetime

import cv2
import face_recognition
import numpy as np
from PIL import Image

IMAGES_FOLDER = "ImagesAttendance"
ATTENDANCE_FILE = "Attendance.csv"
MATCH_THRESHOLD = 0.5  # lower = stricter matching


def load_known_faces(folder):
    """Read every photo in the folder and turn each face into numbers."""
    encodings, names = [], []
    for file_name in os.listdir(folder):
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"Skipping {file_name} (not a jpg/png photo)")
            continue

        # Force the photo into plain 8-bit RGB, which face_recognition needs
        image = Image.open(os.path.join(folder, file_name)).convert("RGB")
        image = np.ascontiguousarray(np.array(image, dtype=np.uint8))

        found = face_recognition.face_encodings(image)
        if not found:
            print(f"No face found in {file_name}, skipping it")
            continue
        encodings.append(found[0])
        clean_name = file_name.split(".")[0].strip()
        names.append(clean_name)
    return encodings, names
    


def mark_attendance(name):
    """Save the name and time in the CSV file, only once per person per day."""
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
        print(f"Attendance marked for {name}")


def main():
    known_encodings, known_names = load_known_faces(IMAGES_FOLDER)
    if not known_encodings:
        print("No faces loaded. Add photos to the ImagesAttendance folder.")
        return
    print(f"Loaded {len(known_names)} known face(s): {known_names}")

    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read from the webcam")
            break

        # Shrink the picture so it runs faster
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, locations)

        for encoding, (top, right, bottom, left) in zip(encodings, locations):
            distances = face_recognition.face_distance(known_encodings, encoding)
            best = int(np.argmin(distances))

            if distances[best] < MATCH_THRESHOLD:
                name = known_names[best]
                mark_attendance(name)
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            # Scale the box back up to the full-size picture
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
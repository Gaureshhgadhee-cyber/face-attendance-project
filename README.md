# Face Recognition Attendance System

A real-time attendance system that uses facial recognition through a webcam to automatically identify people and log their attendance with a timestamp.

## Demo

[Click here to view demo screenshots](DEMO.md)

## Features
- Real-time face detection and recognition via webcam
- Automatically logs name + date + time to Attendance.csv
- Prevents duplicate attendance entries for the same day
- Simple to add new people — just drop a labeled photo into a folder

## Tech Stack
- Python
- OpenCV
- face_recognition (dlib-based)
- NumPy, Pandas

## How to Run
1. Clone this repo
2. Create a virtual environment and activate it
3. Install dependencies: `pip install -r requirements.txt`
   - Note: dlib may need a prebuilt wheel on Windows if pip install fails
4. Add photos of known people to the `ImagesAttendance` folder (filename = person's name)
5. Run: `python main.py`
6. Press `q` to quit

## Future Improvements
- Web interface using Streamlit
- Database instead of CSV
- Anti-spoofing / liveness detection
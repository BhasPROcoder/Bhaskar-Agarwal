import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENTS_DIR = os.path.join(BASE_DIR, "Students")
LOGS_DIR = os.path.join(BASE_DIR, "Logs")
ATTENDANCE_FILE = os.path.join(LOGS_DIR, "attendance.csv")

FRAME_SCALE = 0.25


# -----------------------------
# Load known student faces
# -----------------------------

known_face_encodings = []
known_face_names = []

for filename in os.listdir(STUDENTS_DIR):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image_path = os.path.join(STUDENTS_DIR, filename)

        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"No face found in {filename}. Skipping...")
            continue

        if len(encodings) > 1:
            print(f"Multiple faces found in {filename}. Skipping...")
            continue

        known_face_encodings.append(encodings[0])

        # Use the filename as the student's name
        name = os.path.splitext(filename)[0]
        name = name.replace("_", " ")

        known_face_names.append(name)


print(f"Loaded {len(known_face_names)} students.")


# -----------------------------
# Create Logs folder
# -----------------------------

os.makedirs(LOGS_DIR, exist_ok=True)


# -----------------------------
# Create attendance file
# -----------------------------

if not os.path.exists(ATTENDANCE_FILE):

    with open(ATTENDANCE_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Name",
            "Date",
            "Time"
        ])


# -----------------------------
# Load today's attendance
# -----------------------------

marked_today = set()

today = datetime.now().strftime("%Y-%m-%d")

with open(ATTENDANCE_FILE, "r", newline="") as file:

    reader = csv.reader(file)
    next(reader, None)

    for row in reader:

        if len(row) >= 2 and row[1] == today:
            marked_today.add(row[0])


# -----------------------------
# Function to mark attendance
# -----------------------------

def mark_attendance(name):

    # Don't mark the same student twice
    if name in marked_today:
        return

    current_time = datetime.now().strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            today,
            current_time
        ])

    marked_today.add(name)

    print(f"Attendance marked: {name}")


# -----------------------------
# Start webcam
# -----------------------------

video_capture = cv2.VideoCapture(0)


while True:

    # Grab a frame from the webcam
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to access webcam.")
        break


    # -----------------------------
    # Resize frame for faster processing
    # -----------------------------

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=FRAME_SCALE,
        fy=FRAME_SCALE
    )


    # Convert BGR to RGB
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------
    # Detect faces
    # -----------------------------

    face_locations = face_recognition.face_locations(
        rgb_small_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )


    face_names = []


    # -----------------------------
    # Compare detected faces
    # -----------------------------

    for face_encoding in face_encodings:

        name = "Unknown"

        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding
        )

        # Calculate the distance between faces
        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )


        if len(face_distances) > 0:

            # Find the closest known face
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]


        face_names.append(name)


    # -----------------------------
    # Display results
    # -----------------------------

    for (top, right, bottom, left), name in zip(
        face_locations,
        face_names
    ):

        # Convert coordinates back to original frame size
        top = int(top / FRAME_SCALE)
        right = int(right / FRAME_SCALE)
        bottom = int(bottom / FRAME_SCALE)
        left = int(left / FRAME_SCALE)


        # Draw rectangle around face
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 0, 255),
            2
        )


        # Draw name background
        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            (0, 0, 255),
            cv2.FILLED
        )


        # Display name
        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (255, 255, 255),
            1
        )


        # Mark attendance for recognized students
        if name != "Unknown":
            mark_attendance(name)


    # -----------------------------
    # Display webcam
    # -----------------------------

    cv2.imshow(
        "Face Recognition Attendance",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Release resources
# -----------------------------

video_capture.release()
cv2.destroyAllWindows()

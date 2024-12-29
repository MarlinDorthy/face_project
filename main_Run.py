import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime

# Create the main window
window = tk.Tk()
window.title("Attendance Management System using Face Recognition")
window.geometry("800x600")
window.configure(background='lightblue')  # Set a light blue background for the window

# Create a folder to store student images
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Create a folder for the trained model
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Create the recognizer and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Title Label
title_label = tk.Label(window, text="Attendance Management System using Face Recognition", bg="black", fg="white",
                       font=("Times New Roman", 20, "bold"), pady=10)
title_label.pack(side=tk.TOP, fill=tk.X)

# Enrollment Label and Entry
enrollment_label = tk.Label(window, text="Enter Enrollment:", bg='lightblue', font=("Arial", 14))
enrollment_label.place(x=50, y=100)
enrollment_entry = tk.Entry(window, font=("Arial", 14), width=20)
enrollment_entry.place(x=200, y=100)

# Name Label and Entry
name_label = tk.Label(window, text="Enter Name:", bg='lightblue', font=("Arial", 14))
name_label.place(x=50, y=150)
name_entry = tk.Entry(window, font=("Arial", 14), width=20)
name_entry.place(x=200, y=150)

# Flags to prevent continuous popups
image_taken_flag = False
names = {}  # Store names associated with enrollment numbers
training_flag = False  # Flag to ensure that model is not retrained multiple times

# Buttons for actions (Take Images, Train Images, etc.)
def take_images():
    global image_taken_flag
    # Open the camera
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        # Capture faces
        for (x, y, w, h) in faces:
            enrollment = enrollment_entry.get()
            name = name_entry.get()
            
            if enrollment and name:
                face_id = f"{enrollment}_{name}"
                img_path = os.path.join('dataset', f"{face_id}.jpg")
                cv2.imwrite(img_path, gray[y:y+h, x:x+w])
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Only show the message once
                if not image_taken_flag:
                    messagebox.showinfo("Info", "Images Taken Successfully")
                    image_taken_flag = True
            else:
                messagebox.showwarning("Input Error", "Please enter Enrollment and Name")
                break

        cv2.imshow("Taking Images", img)
        
        # Break after capturing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def upload_image():
    # Open a file dialog to upload an image
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        enrollment = enrollment_entry.get()
        name = name_entry.get()
        
        if enrollment and name:
            face_id = f"{enrollment}_{name}"
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img_array = np.array(img)
            img_path = os.path.join('dataset', f"{face_id}.jpg")
            cv2.imwrite(img_path, img_array)
            messagebox.showinfo("Image Uploaded", "Image uploaded successfully")
        else:
            messagebox.showwarning("Input Error", "Please enter Enrollment and Name")

def train_images():
    global training_flag
    if training_flag:
        return  # Prevent retraining if already trained

    # Training the model
    faces, labels = [], []
    for filename in os.listdir('dataset'):
        if filename.endswith('.jpg'):
            img_path = os.path.join('dataset', filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img_array = np.array(img)
            faces.append(img_array)
            face_id = filename.split('_')[0]  # Extract enrollment number
            labels.append(int(face_id))
            names[face_id] = filename.split('_')[1]  # Store name with the enrollment number

    # Train the recognizer with faces and labels
    if faces and labels:
        recognizer.train(faces, np.array(labels))
        recognizer.save('trainer/trainer.yml')
        messagebox.showinfo("Model Trained", "Model Training Successful")
        training_flag = True  # Set flag after training
    else:
        messagebox.showwarning("Training Error", "No images found to train the model")

def automatic_attendance():
    # Recognize faces and mark attendance
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_id, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 100:  # Confidence threshold (can adjust)
                enrollment = str(face_id)
                name = names.get(enrollment, "Unknown")  # Fetch the name from names dictionary
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{enrollment} {name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Output to the console
                time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Attendance marked for {name} (Enrollment: {enrollment}) at {time_now}")
            else:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Automatic Attendance", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def manually_fill_attendance():
    # Create new window for manual attendance
    manual_window = tk.Toplevel(window)
    manual_window.title("Manual Attendance")
    manual_window.geometry("500x300")  # Adjusted size of the window
    manual_window.configure(background="lightyellow")  # Set background color for manual window

    # Subject Label and Entry
    subject_label = tk.Label(manual_window, text="Enter Subject:", font=("Arial", 14), bg="lightyellow")
    subject_label.place(x=50, y=50)  # Adjusted position
    subject_entry = tk.Entry(manual_window, font=("Arial", 14), width=20)
    subject_entry.place(x=200, y=50)  # Adjusted position

    # Student Name Label and Entry
    student_name_label = tk.Label(manual_window, text="Student Name:", font=("Arial", 14), bg="lightyellow")
    student_name_label.place(x=50, y=100)  # Adjusted position
    student_name_entry = tk.Entry(manual_window, font=("Arial", 14), width=20)
    student_name_entry.place(x=200, y=100)  # Adjusted position

    def mark_attendance():
        subject = subject_entry.get()
        student_name = student_name_entry.get()
        if subject and student_name:
            # Get the current time
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Print the attendance details to the console
            print(f"Attendance marked for {student_name} in {subject} at {time_now}")
        else:
            messagebox.showwarning("Input Error", "Please enter both subject and student name")

    # Button to mark attendance
    mark_button = tk.Button(manual_window, text="Mark Attendance", font=("Arial", 12, "bold"), width=20, height=2, command=mark_attendance, bg="lightgreen", activebackground="darkgreen")
    mark_button.place(x=150, y=150)  # Adjusted position

    # Exit button to close the manual window
    exit_button = tk.Button(manual_window, text="Exit", font=("Arial", 12, "bold"), width=20, height=2, command=manual_window.destroy, bg="lightcoral", activebackground="darkred")
    exit_button.place(x=150, y=200)  # Adjusted position

def exit_system():
    window.quit()

# Grid layout for buttons to align them neatly
button_frame = tk.Frame(window, bg="lightblue")
button_frame.place(x=50, y=200)

# Define button styles
button_style = {
    'font': ("Arial", 14, "bold"),
    'width': 20,
    'height': 2,
    'bg': 'lightgreen',
    'activebackground': 'darkgreen',
    'fg': 'black'
}

take_image_button = tk.Button(button_frame, text="Take Images", command=take_images, **button_style)
take_image_button.grid(row=0, column=0, padx=10, pady=10)

upload_image_button = tk.Button(button_frame, text="Upload Image", command=upload_image, **button_style)
upload_image_button.grid(row=0, column=1, padx=10, pady=10)

train_button = tk.Button(button_frame, text="Train Model", command=train_images, **button_style)
train_button.grid(row=1, column=0, padx=10, pady=10)

attendance_button = tk.Button(button_frame, text="Automatic Attendance", command=automatic_attendance, **button_style)
attendance_button.grid(row=1, column=1, padx=10, pady=10)

manual_button = tk.Button(button_frame, text="Manual Attendance", command=manually_fill_attendance, **button_style)
manual_button.grid(row=2, column=0, padx=10, pady=10)

exit_button = tk.Button(button_frame, text="Exit", command=exit_system, **button_style)
exit_button.grid(row=2, column=1, padx=10, pady=10)

# Run the main loop of the window
window.mainloop()

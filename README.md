# Handwritten Digit Recognizer

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-red)

A **CNN-based handwritten digit recognition app** using Keras and Tkinter.  
Draw a digit on the canvas, and the model predicts it in real time.

## 🧠 Features
- Train your own CNN model on MNIST  
- Recognize handwritten digits using a simple GUI  
- Real-time prediction with accuracy percentage  

## 🚀 Usage
```bash
git clone https://github.com/nimraanis/Handwritten-Digit-Recognizer.git
cd Handwritten-Digit-Recognizer
pip install tensorflow keras numpy pillow pywin32
python train_model.py   # trains and saves mnist.h5
python digit_app.py     # runs the GUI

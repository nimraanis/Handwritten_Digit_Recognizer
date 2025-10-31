from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

# Load trained model
model = load_model('mnist.h5')

def predict_digit(img):
    # Resize and preprocess image
    img = img.resize((28, 28)).convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = (255 - img) / 255.0

    # Predict digit
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.x = self.y = 0

        # Canvas and buttons
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Draw a digit", font=("Helvetica", 24))
        self.classify_btn = tk.Button(self, text="Recognize", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Layout
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        # Bind drawing
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.label.configure(text="Draw a digit")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        im = ImageGrab.grab(rect)
        digit, acc = predict_digit(im)
        self.label.configure(text=f"{digit}, {int(acc*100)}%")

    def draw_lines(self, event):
        r = 8
        self.canvas.create_oval(event.x - r, event.y - r,
                                event.x + r, event.y + r,
                                fill='black')

app = App()
app.mainloop()

import tkinter as tk
import subprocess
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt

class MyGUI:
    def __init__(self):
        self.intro = """  Welcome! This is a ML algorithm visualizer!\n
        The following algorithms have been implemented in python:\n
        1. K nearest neighbours (KNN) : Iris dataset
        2. Linear Regression (LinearReg) : sklearn.datasets(make regression)
        3. Logistic Regression (LogisticReg) : breast cancer dataset
        4. Decision Trees (DecisionTrees) : breast cancer dataset
        5. Random Forest (RandomForest) : breast cancer dataset
        6. Support Vector Machines(SVMs) : sklearn.datasets(make blobs)
          
        Select any algorithm (given in bracket) to see the implementation results:
        """

        self.root = tk.Tk()
        self.root.geometry("800x800")
        self.root.title("ML Algorithms")
        self.root.config(bg="blue")

        self.label = tk.Label(self.root, text=self.intro, font=("Arial", 16))
        self.label.config(justify="left")
        self.label.pack(padx=10, pady=10)

        self.mytext = tk.Text(self.root, height=5, font=("Arial", 16))
        self.mytext.pack(padx=10)

        btn1 = self.button("Select", self.get_val)
        btn2 = self.button("Clear", self.clear)

        self.result_text = tk.Text(
            self.root, height=5, font=("Arial", 16), state=tk.DISABLED
        )
        self.result_text.pack(padx=10)

        self.start()

    def button(self, text="", command=None):
        self.mybutton = tk.Button(
            self.root, text=text, font=("Arial", 16), command=command
        )
        self.mybutton.pack(padx=10, pady=10)

    def get_val(self):
        algo_choice = self.mytext.get("1.0", tk.END).strip()
        if algo_choice:
            command = ["python", f"train_{algo_choice}.py"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, f"\n{result.stdout}\n")
            self.result_text.config(state=tk.DISABLED)
        else:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(
                tk.END, "Error! Please select a valid algorithm!"
            )
            self.result_text.config(state=tk.DISABLED)
        
    def clear(self):
        self.mytext.delete("1.0", tk.END)
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state=tk.DISABLED)

    def start(self):
        self.root.mainloop()

GUI = MyGUI()

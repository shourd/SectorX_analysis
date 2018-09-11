from tkinter import filedialog, Tk

root = Tk()
root.fileName = filedialog.askdirectory()

print(root.fileName)
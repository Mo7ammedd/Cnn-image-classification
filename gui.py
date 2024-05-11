# ****************************************#
#        importing library for splash     #
# ****************************************#

from tkinter import *
from tkinter import font
from PIL import ImageTk, Image
import time
import Model
w = Tk()
width_of_window = 427
height_of_window = 250
screen_width = w.winfo_screenwidth()
screen_height = w.winfo_screenheight()
x_coordinate = (screen_width/2)-(width_of_window/2)
y_coordinate = (screen_height/2)-(height_of_window/2)
w.geometry("%dx%d+%d+%d" %
           (width_of_window, height_of_window, x_coordinate, y_coordinate))
w.overrideredirect(1)  # for hiding titlebar



# ******************************#
#        Main Window            #
# *****************************#
def new_win():
    # ******************************#
    #        Main Window            #
    # *****************************#
    import tkinter as tk
    from tkinter import filedialog
    from PIL import ImageTk, Image
    import numpy
    from tensorflow.keras.models import load_model

  

    model = load_model('Model\\model.h5')
    classes = {
        0: 'airplane',
        1: 'car',
        2: 'bird',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck',
    }

    def upload_image():
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(
            ((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        lable.configure(text=' ')
        show_classify_button(file_path)

    def show_classify_button(file_path):
        classify_btn = Button(top, text="Classify Image",
                              command=lambda: classify(file_path), padx=10, pady=5)
        classify_btn.configure(background="#3498db", foreground="white", font=('arial', 10, 'bold'))
        classify_btn.place(relx=0.79, rely=0.46)

    def classify(file_path):
        image = Image.open(file_path)
        image = image.resize((32, 32))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        pred = int(numpy.argmax(model.predict(image), axis=-1)[0])
        sign = classes[pred]
        print(sign)
        lable.configure(foreground='#3498db', text=sign)

    def print_Accuracy():
       from tensorflow.keras.datasets import cifar10
       (x_train, y_train), (x_test, y_test) = cifar10.load_data()
       x_test = x_test / 255.0
       model = load_model('Model\\model.h5')
       test_accuracy = model.evaluate(x_test, y_test)[1] * 100
       print("Model Final Accuracy:", test_accuracy)
       accuracy_label = Label(top, text=f"Model Accuracy: {test_accuracy:.2f}%", font=('arial', 10, 'bold'))
       accuracy_label.pack()
  

    def center_window(top, width, height):
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()

        # Calculate the x and y coordinates to position the window in the center
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)

        # Set the window size and position
        top.geometry(f"{width}x{height}+{x}+{y}")

    width = 800  # New width
    height = 600  # New height

# ******************************#
#        GUI Main Window       #
# *****************************#
    # GUI
    top = tk.Tk()
    top.iconbitmap("Assets/ai.ico")
    top.geometry('800x600')
    # top.eval('tk::PlaceWindow. center')
    center_window(top, width, height)
    top.title("Image Classification CIFAR10")
    top.configure(background="#f0f0f0")

    # Set Heading
    heading = Label(top, text="Image Classification Using Cnn",
                    pady=20, font=('Game Of Squids', 20, 'bold'))
    heading.configure(background="#f0f0f0", foreground='#3498db')
    heading.pack()


    upload = Button(top, text="Upload Image Here",
                    command=upload_image, padx=10, pady=5)
    upload.configure(background="#3498db", foreground='white',
                     font=('arial', 10, 'bold'))
    upload.pack(side=BOTTOM, pady=50)

    exitt = Button(top, text="       Close       ",
                   command=top.destroy, padx=10, pady=5)
    exitt.configure(background="#3498db", foreground='white',
                    font=('arial', 10, 'bold'))
    exitt.pack(side=BOTTOM, pady=60)
    exitt.place(relx=0.79, rely=0.60)

    btn_arr = Button(top, command=print_Accuracy, text="Show Accuracy", padx=10, pady=5)
    btn_arr.configure(background="#3498db", foreground="white", font=('arial', 10, 'bold'))
    btn_arr.pack(side=BOTTOM, pady=50)
    btn_arr.place(relx=0.120, rely=0.40)
    # upload image
    sign_image = Label(top, background="#f0f0f0")
    sign_image.pack(side=BOTTOM, expand=True)

    # bannerimage
    # Replace with your image path
    path1 = "Assets/Upload_photo.png"
    image1 = Image.open(path1)
    small_image = image1.resize((100, 100), Image.LANCZOS)

    # Convert the image to Tkinter format
    tk_imagee = ImageTk.PhotoImage(small_image)

    # Create a label and display the image
    label = tk.Label(top, background="#f0f0f0", image=tk_imagee)
    label.pack(pady=10)
    label.pack()

    # clsass
    lable = Label(top, background="#f0f0f0", font=('arial', 15, 'bold'))
    lable.pack(side=BOTTOM, expand=True)

    arr = Label(top, text=" " ,pady=10, font=('Game Of Squids', 20, 'bold'))
    arr.configure(background="#f0f0f0", foreground='#3498db')
    arr.pack()
    arr.place(relx = 0.150 , rely = 0.60)
    top.mainloop()


Frame(w, width=427, height=250, bg='#3498db').place(x=0, y=0)
label1 = Label(w, text='PROGRAMEDD', fg='#ffffff',
               bg='#3498db')  # decorate it
# You need to install this font in your PC or try another one
label1.configure(font=('Game Of Squids', 24, "bold"))
label1.place(x=80, y=90)

label2 = Label(w, text='Loading...', fg='#cccccc',
               bg='#3498db')  # decorate it
label2.configure(font=("Calibri", 11))
label2.place(x=10, y=215)

# making animation

image_a = ImageTk.PhotoImage(Image.open(
    'Assets/c2.png'))
image_b = ImageTk.PhotoImage(Image.open(
    'Assets/c1.png'))

for i in range(5):  # 5loops
    l1 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=180, y=145)
    l2 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=200, y=145)
    l3 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=220, y=145)
    l4 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=240, y=145)
    w.update_idletasks()
    time.sleep(0.5)

    l1 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=180, y=145)
    l2 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=200, y=145)
    l3 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=220, y=145)
    l4 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=240, y=145)
    w.update_idletasks()
    time.sleep(0.5)

    l1 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=180, y=145)
    l2 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=200, y=145)
    l3 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=220, y=145)
    l4 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=240, y=145)
    w.update_idletasks()
    time.sleep(0.5)

    l1 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=180, y=145)
    l2 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=200, y=145)
    l3 = Label(w, image=image_b, border=0, relief=SUNKEN).place(x=220, y=145)
    l4 = Label(w, image=image_a, border=0, relief=SUNKEN).place(x=240, y=145)
    w.update_idletasks()
    time.sleep(0.5)
w.destroy()
new_win()
w.mainloop()
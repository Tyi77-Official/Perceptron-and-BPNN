from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import train_test_split
from random import *

import API
import main

def train(*args):
        '''Use single-layer perceptron to train the module'''
        # ---Feth the data and some settings---
        try:
            file = args[0].get()
            data = np.loadtxt(f'Basic/{file}')
        except:
            return

        data_theta = np.insert(data, 0, -1.0, axis=1) # Add the set of -1 to the first column

        train_data, test_data = train_test_split(data_theta, random_state=777, train_size=2/3) # Split

        train_data_modified = np.hsplit(train_data, np.array([3]))
        train_data_x = train_data_modified[0]
        train_data_d = train_data_modified[1].squeeze()
        train_data_n = np.shape(train_data_x)[0]
        train_data_y = np.empty(train_data_n)

        test_data_modified = np.hsplit(test_data, np.array([3]))
        test_data_x = test_data_modified[0]
        test_data_d = test_data_modified[1].squeeze()
        test_data_n = np.shape(test_data_x)[0]
        test_data_y = np.empty(test_data_n)

        lr = args[1].get()

        try:
            epoch_num = int(args[2].get())
        except:
            return

        # ---Training---
        w = np.repeat(-1 + 2 * random(), 3) # Initialize the weight [-1, 1)
        if file == 'perceptron1.txt' or file == 'perceptron2.txt':
            for t in range(epoch_num): # Batch
                accuracy_count = 0
                for i in range(train_data_n): # Pattern
                    x = train_data_x[i]
                    d = train_data_d[i]
                    v = np.matmul(w, x)

                    if v >= 0:
                        y = 1.0
                    else:
                        y = 0.0
                    train_data_y[i] = y

                    if y > d:
                        w = w - lr * x
                    elif y < d:
                        w = w + lr * x
                    else:
                        accuracy_count += 1
                if accuracy_count == train_data_n:
                    break
        else:
            for t in range(epoch_num): # Batch
                accuracy_count = 0
                for i in range(train_data_n): # Pattern
                    x = train_data_x[i]
                    d = train_data_d[i]
                    v = np.matmul(w, x)

                    if v >= 0:
                        y = 2.0
                    else:
                        y = 1.0
                    train_data_y[i] = y

                    if y > d:
                        w = w - lr * x
                    elif y < d:
                        w = w + lr * x
                    else:
                        accuracy_count += 1
                if accuracy_count == train_data_n:
                    break
        # Calculate Training Accuracy
        training_accuracy_count = 0
        for i in range(train_data_n):
            if train_data_y[i] == train_data_d[i]:
                training_accuracy_count += 1

        training_accuracy_ratio = training_accuracy_count / train_data_n
        args[3].config(text=str(round(training_accuracy_ratio, 3)))

        # ---Disply Weight---
        args[4].config(text='(' + ', '.join(str(round(e, 3)) for e in w) + ')')
        
        # ---Testing---
        testing_accuracy_count = 0
        if file == 'perceptron1.txt' or file == 'perceptron2.txt':
            for i in range(test_data_n):
                v = np.matmul(w, test_data_x[i])
                
                if v >= 0:
                    y = 1.0
                else:
                    y = 0.0
                test_data_y[i] = y
                
                if y == test_data_d[i]:
                    testing_accuracy_count += 1
        else:
            for i in range(test_data_n):
                v = np.matmul(w, test_data_x[i])
                
                if v >=0:
                    y = 2.0
                else:
                    y = 1.0
                test_data_y[i] = y
                
                if y == test_data_d[i]:
                    testing_accuracy_count += 1
        # Calculate Testing Accuracy
        testing_accuracy_ratio = testing_accuracy_count / test_data_n
        args[5].config(text=str(round(testing_accuracy_ratio, 3)))

        # ---Create the line---
        axis1_max = np.hsplit(data, [1, 2])[0].max()
        axis1_min = np.hsplit(data, [1, 2])[0].min()
        axis2_max = np.hsplit(data, [1, 2])[1].max()
        axis2_min = np.hsplit(data, [1, 2])[1].min()
        
        axis1 = np.linspace(axis1_min, axis1_max, 1000)
        axis2 = (w[0] - w[1] * axis1) / w[2]
        axis2 = np.ma.masked_greater(axis2, axis2_max)
        axis2 = np.ma.masked_less(axis2, axis2_min)

        # ---Plot---
        fig = Figure(figsize=(10, 10))
        fig2 = Figure(figsize=(10, 10))
        a = fig.add_subplot(111)
        b = fig2.add_subplot(111)

        a.scatter([i[1] if i[3] == 0. else None for i in train_data], [i[2] if i[3] == 0. else None for i in train_data], color='red', s=8)
        a.scatter([i[1] if i[3] == 1. else None for i in train_data], [i[2] if i[3] == 1. else None for i in train_data], color='green', s=8)
        a.scatter([i[1] if i[3] == 2. else None for i in train_data], [i[2] if i[3] == 2. else None for i in train_data], color='blue', s=8)

        b.scatter([i[1] if i[3] == 0. else None for i in test_data], [i[2] if i[3] == 0. else None for i in test_data], color='red', s=8)
        b.scatter([i[1] if i[3] == 1. else None for i in test_data], [i[2] if i[3] == 1. else None for i in test_data], color='green', s=8)
        b.scatter([i[1] if i[3] == 2. else None for i in test_data], [i[2] if i[3] == 2. else None for i in test_data], color='blue', s=8)

        a.set_title('Training')
        b.set_title('Testing')

        a.plot(axis1, axis2, '#ff00ff', linewidth=1)
        b.plot(axis1, axis2, '#ff00ff', linewidth=1)

        train_canvas = FigureCanvasTkAgg(fig, master=args[6])
        train_canvas.get_tk_widget().place(x=470, y=20, height=349, width=349)
        # train_canvas.draw()

        test_canvas = FigureCanvasTkAgg(fig2, master=args[6])
        test_canvas.get_tk_widget().place(x=470, y=370, height=349, width=349)
        # test_canvas.draw()

def basic_frame_content():
    # ---Fetch globals properties from API.py---
    root = API.root
    basic_frame = API.basic_frame
    mjh1 = API.mjh1
    mjh2 = API.mjh2
    mjh3 = API.mjh3
    # ---Set the widgets---
    title = Label(basic_frame, text='Basic', font=mjh1)

    fileName = StringVar()
    fileName_list = ['2Ccircle1.txt', '2Circle1.txt', '2CloseS.txt', '2CloseS2.txt', '2CloseS3.txt', '2cring.txt', '2CS.txt', '2Hcircle1.txt', '2ring.txt', 'perceptron1.txt', 'perceptron2.txt']
    file_option = OptionMenu(basic_frame, fileName, *fileName_list)
    file_option.config(font=mjh2)

    learning_rate_lb = Label(basic_frame, text='Learning Rate', font=mjh3)
    learning_rate = Scale(basic_frame, from_=0.0, to=1.0, resolution=0.01, tickinterval=0.5, orient='horizontal', cursor='cross')

    epoch_lb = Label(basic_frame, text='Epoch', font=mjh3)
    epoch = Entry(basic_frame, font=mjh3)

    training_accuracy_lb = Label(basic_frame, text='Training Accuracy', font=mjh3)
    training_accuracy = Label(basic_frame, text='...', font=mjh3, anchor=CENTER)

    testing_accuracy_lb = Label(basic_frame, text='Testing Accuracy', font=mjh3)
    testing_accuracy = Label(basic_frame, text='...', font=mjh3, anchor=CENTER)

    weight_lb = Label(basic_frame, text='Weight(Threshold, w...)', font=mjh3)
    weight = Label(basic_frame, text='...', font=mjh3, anchor=CENTER)

    train_graph = Canvas(basic_frame, relief='ridge', borderwidth='2')
    test_graph = Canvas(basic_frame, relief='ridge', borderwidth='2')

    args = [fileName, learning_rate, epoch, training_accuracy, weight, testing_accuracy, basic_frame]
    submit = Button(basic_frame, text='Training!', font=mjh3, command=lambda: train(*args))

    exitBt = Button(basic_frame, text='Exit!', font=mjh3, command=root.quit)

    # ---Place widgets---
    title.place(x=110, y=30, height=65, width=264)

    file_option.place(x=100, y=110, height=50, width=300)

    learning_rate_lb.place(x=70, y=190, height=35, width=165)
    learning_rate.place(x=260, y=180, height=66, width=145)
    epoch_lb.place(x=110, y=269, height=25, width=77)
    epoch.place(x=285, y=260, height=41, width=95)

    submit.place(x=180, y=340, height=41, width=120)

    training_accuracy_lb.place(x=50, y=440, height=35, width=199)
    training_accuracy.place(x=335, y=440, height=35, width=50)
    testing_accuracy_lb.place(x=60, y=500, height=35, width=188)
    testing_accuracy.place(x=335, y=500, height=35, width=50)
    weight_lb.place(x=30, y=560, height=35, width=250)
    weight.place(x=260, y=560, height=35, width=200)

    train_graph.place(x=470, y=20, height=350, width=350)
    test_graph.place(x=470, y=370, height=350, width=350)

    exitBt.place(x=300, y=620, height=41, width=120)


if __name__ == '__main__':
    main.start_up()
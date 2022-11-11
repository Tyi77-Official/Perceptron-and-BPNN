from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import train_test_split
from random import *

import API
import main # Go to the right entering code.

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_diff(x: float) -> float:
    return 1 / (1 + np.exp(-x)) * (1 - 1 / (1 + np.exp(-x)))

def reLU(x: float) -> float:
    return max(0, x)

def reLU_diff(x: float) -> float:
    return 1

def normalize(x: float, min: float, max: float, floor: float, ceiling: float) -> float:
    return floor + (ceiling - floor) * (x - min) / (max - min)

def train(*args):
    '''args = [fileName, FILE_CONSTRUCTION, rbVariable, learning_rate, epoch, training_accuracy, testing_accuracy, advanced_frame, progressbar, fig, train_canvas, fig2, test_canvas, weight]'''
    # ---Feth the data and some settings---
    try:
        file = args[0].get()
        data = np.loadtxt(f'Bonus/{file}')
        for e in args[1]:
            if file == e['name']:
                input_dim = e['input_dim']
                hidden_dim = e['hidden_dim']
                output_dim = e['output_dim']
                if args[2].get() == 1:
                    activation_func = globals()['sigmoid']
                    activation_func_diff = globals()['sigmoid_diff']
                elif args[2].get() == 2:
                    activation_func = globals()['reLU']
                    activation_func_diff = globals()['reLU_diff']
    except:
        return

    # Split
    train_data, test_data = train_test_split(data, random_state=777, train_size=2/3)

    # Extract x, d, y from the training data and also the testing data
    train_data_modified = np.hsplit(train_data, np.array([input_dim]))
    train_data_x = train_data_modified[0]
    train_data_d = train_data_modified[1].squeeze()
    train_data_n = np.shape(train_data_x)[0]
    

    test_data_modified = np.hsplit(test_data, np.array([input_dim]))
    test_data_x = test_data_modified[0]
    test_data_d = test_data_modified[1].squeeze()
    test_data_n = np.shape(test_data_x)[0]

    # Fetch learning rate
    lr = args[3].get()

    # Fetch epoch
    try:
        epoch_num = int(args[4].get())
    except:
        return

    # ---Encoding---
    d_unique = np.unique(np.concatenate((train_data_d, test_data_d)))
    d_unique_size = d_unique.shape[0]
    d_o = []
    for i in train_data_d:
        idx = np.where(d_unique==i)[0][0]
        tmp = np.zeros(d_unique_size, dtype=float)
        tmp[idx] = 1
        d_o.append(tmp)
    d_o = np.array(d_o)

    # ---Initializing---
    # Initialize the weight between the input layer and the hidden layer [-1, 1)
    w_ih = np.full([hidden_dim, input_dim], -1. + 2. * random())
    # Initialize the weight between the hidden layer and the output layer [-1, 1)
    w_ho = np.full([output_dim, hidden_dim], -1. + 2. * random())
    # Initialize the threshold of the hidden layer and the outputlayer [-50, 50)
    b_h = np.repeat(-50. + 100. * random(), hidden_dim)
    b_o = np.repeat(-50. + 100. * random(), output_dim)

    # Initialize multiple y and delta
    y_h = np.empty(hidden_dim) # Ouput from the hidden layer
    y_diff_h = np.empty(hidden_dim)
    y_o = np.empty(output_dim) # Output from the output layer
    y_diff_o = np.empty(output_dim)
    delta_o = np.empty(output_dim)
    delta_h = np.empty(hidden_dim)
    accuracy_record = np.empty(train_data_n)

    # ---Training---
    for t in range(epoch_num): # Batch
        args[8].config(value=100 * t / epoch_num)
        args[7].update_idletasks()
        accuracy_count = 0
        for i in range(train_data_n): # Pattern
            x = train_data_x[i]
            # Input -> Hidden
            for j in range(hidden_dim):
                v = np.dot(x, w_ih[j]) - b_h[j]
                y_h[j] = activation_func(v)
                y_diff_h[j] = activation_func_diff(v)
            # Hidden -> Output
            for j in range(output_dim):
                v = np.dot(y_h, w_ho[j]) - b_o[j]
                y_o[j] = activation_func(v)
                y_diff_o[j] = activation_func_diff(v)

            # Calculate Accuracy
            if y_o.argmax() == d_o.argmax():
                accuracy_record[i] = 1
                accuracy_count += 1
            else:
                accuracy_record[i] = 0
            if accuracy_count == train_data_n:
                break
            
            # Delta : Output
            for j in range(output_dim):
                delta_o[j] = (d_o[i, j] - y_o[j]) * y_diff_o[j]
            # Delta : Hidden
            for j in range(hidden_dim):
                sum_o = 0
                for k in range(output_dim):
                    sum_o += (d_o[i, k] - y_o[k]) * y_diff_o[k] * w_ho[k, j]
                delta_h[j] = y_diff_h[j] * sum_o
            # Reweight : w_ih, b_h
            for k in range(hidden_dim):
                for j in range(input_dim):
                    w_ih[k, j] += lr * delta_h[k] * train_data_x[i, j]
                b_h[k] -= lr * delta_h[k]
            # Reweight : w_ho, b_o
            for k in range(output_dim):
                for j in range(hidden_dim):
                    w_ho[k, j] += lr * delta_o[k] * y_h[j]
                b_o[k] -= lr * delta_o[k]
    args[8].config(value=0)
    # Calculate Training Accuracy
    training_accuracy_count = np.count_nonzero(accuracy_record)

    training_accuracy_ratio = training_accuracy_count / train_data_n
    args[5].config(text=str(round(training_accuracy_ratio, 3)))
    
    # ---Testing---
    accuracy_count = 0
    for i in range(test_data_n): # Pattern
        x = test_data_x[i]
        # Input -> Hidden
        for j in range(hidden_dim):
            v = np.dot(x, w_ih[j]) - b_h[j]
            y_h[j] = activation_func(v)
            y_diff_h[j] = activation_func_diff(v)
        # Hidden -> Output
        for j in range(output_dim):
            v = np.dot(y_h, w_ho[j]) - b_o[j]
            y_o[j] = activation_func(v)
            y_diff_o[j] = activation_func_diff(v)

        # Calculate Accuracy
        if y_o.argmax() == d_o.argmax():
            accuracy_count += 1
    # Calculate Testing Accuracy
    testing_accuracy_ratio = accuracy_count / test_data_n
    args[6].config(text=str(round(testing_accuracy_ratio, 3)))

    # ---Plot---
    fig = args[9]
    train_canvas = args[10]
    fig2 = args[11]
    test_canvas = args[12]
    if file == '2Circle2.txt' or file == 'perceptron3.txt' or file == 'perceptron4.txt' or file == 'C3D.txt' or file == 'xor.txt':
        if file == '2Circle2.txt' or file == 'perceptron4.txt' or file == 'xor.txt':
            a = fig.add_subplot(111)
            b = fig2.add_subplot(111)

            
            a.scatter([i[0] for i in train_data_x], [i[1] for i in train_data_x], c=train_data_d, s=8)
            b.scatter([i[0] for i in test_data_x], [i[1] for i in test_data_x], c=test_data_d, s=8)            
        else:
            a = fig.add_subplot(111, projection='3d')
            b = fig2.add_subplot(111, projection='3d')

            a.scatter([i[0] for i in train_data_x], [i[1] for i in train_data_x], [i[2] for i in train_data_x], c=train_data_d, s=8)
            b.scatter([i[0] for i in test_data_x], [i[1] for i in test_data_x], [i[2] for i in test_data_x], c=test_data_d, s=8)

        a.set_title('Training')
        b.set_title('Testing')

        train_canvas.get_tk_widget().place(x=470, y=20, height=349, width=349)
        train_canvas.draw()
        test_canvas.get_tk_widget().place(x=470, y=370, height=349, width=349)
        test_canvas.draw()
    else:
        train_canvas.get_tk_widget().place_forget()
        test_canvas.get_tk_widget().place_forget()
        
    # ---Output Weight---
    w_ih_text = np.array2string(w_ih)
    b_h_text = np.array2string(b_h)
    w_ho_text = np.array2string(w_ho)
    b_o_text = np.array2string(b_o)
    with open('weight.txt', mode='w') as wfile:
        wfile.write(f'{file}\n')
        wfile.write('Weights: Input -> Hidden\n')
        wfile.write(w_ih_text+'\n')
        wfile.write('Thresholds: Hidden\n')
        wfile.write(b_h_text+'\n')
        wfile.write('Weights: Hidden -> Output\n')
        wfile.write(w_ho_text+'\n')
        wfile.write('Thresholds: Output\n')
        wfile.write(b_o_text+'\n')
    

def advanced_frame_content():
    # ---Fetch globals properties from API.py---
    root = API.root
    advanced_frame = API.advanced_frame
    mjh1 = API.mjh1
    mjh2 = API.mjh2
    mjh3 = API.mjh3

    # ---Widgets Setting---
    title = Label(advanced_frame, text='Advanced', font=mjh1)

    fileName = StringVar(advanced_frame)
    fileName_list = ['2Circle2.txt', '4satellite-6.txt', '5CloseS1.txt', '8OX.txt', 'C3D.txt', 'C10D.txt', 'IRIS.txt', 'Number.txt', 'perceptron3.txt', 'perceptron4.txt', 'wine.txt', 'xor.txt']
    file_option = OptionMenu(advanced_frame, fileName, *fileName_list)
    file_option.config(font=mjh2)

    rbVariable = IntVar(advanced_frame, value=1)
    sigmoid_rb = Radiobutton(advanced_frame, text='Sigmoid', font=mjh3, variable=rbVariable, value=1)
    reLU_rb = Radiobutton(advanced_frame, text='ReLU', font=mjh3, variable=rbVariable, value=2)

    learning_rate_lb = Label(advanced_frame, text='Learning Rate', font=mjh3)
    learning_rate = Scale(advanced_frame, from_=0.0, to=1.0, resolution=0.01, tickinterval=0.5, orient='horizontal', cursor='cross')

    epoch_lb = Label(advanced_frame, text='Epoch', font=mjh3)
    epoch = Entry(advanced_frame, font=mjh3)

    training_accuracy_lb = Label(advanced_frame, text='Training Accuracy', font=mjh3)
    training_accuracy = Label(advanced_frame, text='...', font=mjh3, anchor=CENTER)

    testing_accuracy_lb = Label(advanced_frame, text='Testing Accuracy', font=mjh3)
    testing_accuracy = Label(advanced_frame, text='...', font=mjh3, anchor=CENTER)

    weight_lb = Label(advanced_frame, text='Weight(Threshold, w...)', font=mjh3)
    weight = Label(advanced_frame, text='...', font=mjh3, anchor=CENTER)

    train_graph = Canvas(advanced_frame, relief='ridge', borderwidth='2')
    test_graph = Canvas(advanced_frame, relief='ridge', borderwidth='2')

    fig = Figure(figsize=(10, 10))
    fig2 = Figure(figsize=(10, 10))

    train_canvas = FigureCanvasTkAgg(fig, master=advanced_frame)
    test_canvas = FigureCanvasTkAgg(fig2, master=advanced_frame)

    # ---Set FILE_CONSTRUCTION---
    FILE_CONSTRUCTION = [
        {'name': '2Circle2.txt', 'input_dim': 2, 'hidden_dim': 5, 'output_dim': 3},
        {'name': '4satellite-6.txt', 'input_dim': 4, 'hidden_dim': 8, 'output_dim': 6},
        {'name': '5CloseS1.txt', 'input_dim': 2, 'hidden_dim': 10, 'output_dim': 2},
        {'name': '8OX.txt', 'input_dim': 8, 'hidden_dim': 2, 'output_dim': 3},
        {'name': 'C3D.txt', 'input_dim': 3, 'hidden_dim': 8, 'output_dim': 4},
        {'name': 'C10D.txt', 'input_dim': 10, 'hidden_dim': 7, 'output_dim': 4},
        {'name': 'IRIS.txt', 'input_dim': 4, 'hidden_dim': 10, 'output_dim': 3},
        {'name': 'Number.txt', 'input_dim': 25, 'hidden_dim': 2, 'output_dim': 10},
        {'name': 'perceptron3.txt', 'input_dim': 3, 'hidden_dim': 5, 'output_dim': 2},
        {'name': 'perceptron4.txt', 'input_dim': 2, 'hidden_dim': 7, 'output_dim': 3},
        {'name': 'wine.txt', 'input_dim': 13, 'hidden_dim': 4, 'output_dim': 3},
        {'name': 'xor.txt', 'input_dim': 2, 'hidden_dim': 2, 'output_dim': 2}
    ]

    progressbar = ttk.Progressbar(advanced_frame, length=160, value=0)
    args = [fileName, FILE_CONSTRUCTION, rbVariable, learning_rate, epoch, training_accuracy, testing_accuracy, advanced_frame, progressbar, fig, train_canvas, fig2, test_canvas, weight]
    submit = Button(advanced_frame, text='Training!', font=mjh3, command=lambda: train(*args))

    exitBt = Button(advanced_frame, text='Exit!', font=mjh3, command=root.quit)

    # ---Place widgets---
    title.place(x=110, y=30, height=65, width=264)

    file_option.place(x=100, y=110, height=50, width=300)

    sigmoid_rb.place(x=100, y=180, height=30, width=100)
    reLU_rb.place(x=280, y=180, height=30, width=100)

    learning_rate_lb.place(x=70, y=230, height=35, width=165)
    learning_rate.place(x=260, y=220, height=66, width=145)
    epoch_lb.place(x=110, y=309, height=25, width=77)
    epoch.place(x=285, y=300, height=41, width=95)

    submit.place(x=90, y=360, height=41, width=120)
    progressbar.place(x=230, y=370, height=20, width=180)

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
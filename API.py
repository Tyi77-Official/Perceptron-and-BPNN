from tkinter import *
from tkinter import font

import main

def define_font():
    global mjh1, mjh2, mjh3
    mjh1 = font.Font(family='Microsoft JhengHei UI', size=34, weight='bold')
    mjh2 = font.Font(family='Microsoft JhengHei UI', size=20)
    mjh3 = font.Font(family='Microsoft JhengHei UI', size=14)

def create_window():
    global root
    root = Tk()
    root.title('HW1')
    # Designate height and width of the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    app_width = 900
    app_height = 750

    x = (screen_width - app_width) // 2
    y = (screen_height - app_height) // 2

    root.geometry(f'{app_width}x{app_height}+{x}+{y}')

def create_frames():
    global basic_frame, advanced_frame
    basic_frame = Frame(root, width=900, height=750)
    advanced_frame = Frame(root, width=900, height=750)

def hide_all_frames():
    basic_frame.pack_forget()
    advanced_frame.pack_forget()

def activate_basic_frame():
    hide_all_frames()
    basic_frame.pack(fill='both', expand=1)

def activate_advanced_frame():
    hide_all_frames()
    advanced_frame.pack(fill='both', expand=1)
    
def create_menu():
    main_menu= Menu(root)
    root.config(menu=main_menu)

    menu_options = Menu(main_menu, activebackground='gray', tearoff=0)
    main_menu.add_cascade(label='Options', menu=menu_options)
    menu_options.add_command(label='Basic', command=activate_basic_frame)
    menu_options.add_separator()
    menu_options.add_command(label='Advanced', command=activate_advanced_frame)

if __name__ == '__main__':
    main.start_up()
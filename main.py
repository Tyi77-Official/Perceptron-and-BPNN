import basic
import advanced
import API

def start_up():
    # Create a new window
    API.create_window()
    root = API.root
    # Create the basic frame and the advanced frame
    API.create_frames()
    basic_frame = API.basic_frame
    advanced_frame = API.advanced_frame
    # Create the Menu
    API.create_menu()
    # Define fonts
    API.define_font()

    # Show the app window
    basic.basic_frame_content()
    advanced.advanced_frame_content()
    basic_frame.pack(fill='both', expand=1)
    # advanced_frame.pack(fill='both', expand=1)
    root.mainloop()

if __name__ == '__main__':
    start_up()
from tkinter import Button

class PlayButton:
    def __init__(self, master, on_play):
        self.on_play_callback = on_play
        self.button = Button(master, text="Play", command=self.on_play)
        # self.button.pack()  # remove this if you want to pack externally

    def pack(self, **kwargs):
        self.button.pack(**kwargs)

    def on_play(self):
        self.on_play_callback()
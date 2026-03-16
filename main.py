import cv2
import numpy as np
import time
import torch

import tkinter as tk
import cv2
from PIL import Image, ImageTk

class CellularAutomataApp:
    def __init__(self, root:tk.Tk,kernel:np.ndarray, start_image:np.ndarray,activation_function:function, interval_ms:int=1,device:torch.device = torch.device("cpu")):
        '''
        A constructor for the cellular automata app.
        
        Parameters:
            root - tk.Tk
                a handle for tkinter
            kernel - np.ndarray
                a 3x3 kernel used by application
            start_image - np.ndarray
                start image pixels should be normalized
                with a value between 0 and 1.
            activation_function - function
                a function used to filter convolution
                output.
            interval_ms - int
                a time in which application is refreshed in 
                miliseconds
            device - torch.device
                a device on which simulation should work.
        '''
        self.root = root
        self.root.title("Celluar automata")
        
        self.interval = interval_ms
        self.activation_function = activation_function
        
        self.canvas_width = 640
        self.canvas_height = 640
        
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<Configure>", self.on_resize)
        
        self.image = torch.tensor(start_image,dtype=torch.float32)
        pil_img = Image.fromarray(self.image.cpu().numpy())
        self.tk_img = ImageTk.PhotoImage(image=pil_img) 
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        self.kernel = torch.tensor(kernel,dtype=torch.float32)[None,None,:,:]
        
        self.cnn = torch.nn.Conv2d(1,1,kernel_size=3,bias=False,padding_mode='reflect',stride=1, padding='same',dilation=1)
                        
        with torch.no_grad():
            self.cnn.weight = torch.nn.Parameter(self.kernel)
                
        self.image = self.image.to(device=device)
        self.cnn = self.cnn.to(device=device)
                
        self.update_image()
        
    def on_resize(self, event):
        # Update our saved dimensions based on the new canvas size
        self.canvas_width = event.width
        self.canvas_height = event.height
        
        self.display_image()
        
    def display_image(self):
        
        img = self.image.cpu().detach().numpy()
        
        resized_img = cv2.resize(img,(self.canvas_width,self.canvas_height))
        # resized_img = cv2.normalize(resized_img,None,alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        resized_img*=255
        pil_img = Image.fromarray(resized_img)
        self.tk_img = ImageTk.PhotoImage(image=pil_img)
        
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_img)    
    
    def update_image(self):
        with torch.no_grad():
            self.image = self.cnn(self.image[None,None,:,:])[0][0]
            self.image = self.activation_function(self.image)
            self.image = torch.clamp(self.image,0,1)
                        
            self.display_image()
        
            self.root.after(self.interval, self.update_image)
        
       
# Moving to the left 
def activation_func(x):
    return x

kernel = np.array([
                [0.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,0.0]
            ])

kernel_1 = np.array([
                [0.1,-0.1,0.3],
                [0,1,-0.3],
                [-0.5,-0.1,0.2]
            ])

def activation_func_1(x):
    
    x = 1 - (2.5/np.sqrt(2*np.pi))*torch.exp(-(x**2)/2)
        
    return x

kernel_2 = np.array([
                [0.68,-0.9,0.68],
                [-0.9,-0.66,-0.9],
                [0.68,-0.9,0.68]
            ])


if __name__ == "__main__":
    
    device = torch.device("cuda")
    
    root = tk.Tk()
    
    image = np.random.random((64,64))
        
    app = CellularAutomataApp(root,kernel_2, image,activation_func_1, interval_ms=10)
    
    root.mainloop()
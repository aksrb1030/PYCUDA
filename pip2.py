from Tkinter import *

from PIL import Image, ImageTk
import time

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy



def blackWhite(inPath , outPath , mode = "luminosity",log = 0):

    if log == 1 :
        print ("----------> SERIAL CONVERSION")
    totalT0 = time.time()

    im = Image.open(inPath)
    px = numpy.array(im)

    getDataT1 = time.time()

    print ("-----> Opening path :" , inPath)

    processT0 =  time.time()
    for x in range(im.size[1]):
        for y in range(im.size[0]):

            r = px[x][y][0]
            g = px[x][y][1]
            b = px[x][y][2]
            if mode == "luminosity" :
                val =  int(0.21 *float(r)  + 0.71*float(g)  + 0.07 * float(b))

            else :
                val = int((r +g + b) /3)

            px[x][y][0] = val
            px[x][y][1] = val
            px[x][y][2] = val

    processT1= time.time()
    #px = numpy.array(im.getdata())
    im = Image.fromarray(px)
    im.save(outPath)

    print ("-----> Saving path :" , outPath)
    totalT1 = time.time()

    # if log == 1 :
    #     print ("Image size : ",im.size)
    #     print ("get and convert Image data  : " ,getDataT1-totalT0 )
    #     print ("Processing data : " , processT1 - processT0 )
    #     print ("Save image time : " , totalT1-processT1)
    #     print ("total  Execution time : " ,totalT1-totalT0 )
    #     print ("\n")

def CudablackWhite(inPath , outPath , mode = "luminosity" , log = 0):

    if log == 1 :
        print ("----------> CUDA CONVERSION")

    totalT0 = time.time()

    im = Image.open(inPath)
    px = numpy.array(im)
    px = px.astype(numpy.float32)

    getAndConvertT1 = time.time()

    allocT0 = time.time()
    d_px = cuda.mem_alloc(px.nbytes)
    cuda.memcpy_htod(d_px, px)

    allocT1 = time.time()

    #Kernel declaration
    kernelT0 = time.time()

    #Kernel grid and block size
    BLOCK_SIZE = 1024
    block = (1024,1,1)
    checkSize = numpy.int32(im.size[0]*im.size[1])
    grid = (int(im.size[0]*im.size[1]/BLOCK_SIZE)+1,1,1)

    #Kernel text
    kernel = """
 
    __global__ void bw( float *inIm, int check ){
 
        int idx = (threadIdx.x ) + blockDim.x * blockIdx.x ;
 
        if(idx *3 < check*3)
        {
        int val = 0.21 *inIm[idx*3] + 0.71*inIm[idx*3+1] + 0.07 * inIm[idx*3+2];
 
        inIm[idx*3]= val;
        inIm[idx*3+1]= val;
        inIm[idx*3+2]= val;
        }
    }
    """

    #Compile and get kernel function
    mod = SourceModule(kernel)
    func = mod.get_function("bw")
    func(d_px,checkSize, block=block,grid = grid)

    kernelT1 = time.time()

    #Get back data from gpu
    backDataT0 = time.time()

    bwPx = numpy.empty_like(px)
    cuda.memcpy_dtoh(bwPx, d_px) # Device -> 'Host
    bwPx = (numpy.uint8(bwPx))

    backDataT1 = time.time()

    #Save image
    storeImageT0 = time.time()
    pil_im = Image.fromarray(bwPx,mode ="RGB")

    pil_im.save(outPath)
    print ("-----> Saving path :" , outPath)

    totalT1 = time.time()

    getAndConvertTime = getAndConvertT1 - totalT0
    allocTime = allocT1 - allocT0
    kernelTime = kernelT1 - kernelT0
    backDataTime = backDataT1 - backDataT0
    storeImageTime =totalT1 - storeImageT0
    totalTime = totalT1-totalT0

    # if log == 1 :
    #     print ("Image size : ",im.size)
    #     print ("get and convert Image data to gpu ready : " ,getAndConvertTime )
    #     print ("allocate mem to gpu: " , allocTime )
    #     print ("Kernel execution time : " , kernelTime)
    #     print ("Get data from gpu and convert : " , backDataTime)
    #     print ("Save image time : " , storeImageTime)
    #     print ("total  Execution time : " ,totalTime )
    #     print ("\n")

class Window(Frame):
    imgPath = 1
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        master.geometry("1200x400")
        self.init_window()
        self.frame = Frame(master, background = 'white')

    def showImg(self):
        Frame.__init__(self)
        self.init_window()
        load = Image.open('720image.jpg')
        load = load.resize((400,300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image =render)
        img.image=render
        img.place(x=0,y=0)

        # status = Label(root, text="showimg...", bd =1,relief= SUNKEN, anchor=W)
        # status.pack(side=BOTTOM, fill=X)
        self.imgPath = '720image.jpg'

    def showImg2(self):
        Frame.__init__(self)
        self.init_window()
        load = Image.open('1Kimage.jpg')
        load = load.resize((400,300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        img = Label(self, image =render)
        img.image=render
        img.place(x=0,y=0)
        self.imgPath = '1Kimage.jpg'
        # status = Label(root, text="showimg...", bd =1,relief= SUNKEN, anchor=W)
        # status.pack(side=BOTTOM, fill=X)

    def showImg3(self):
        Frame.__init__(self)
        self.init_window()
        load = Image.open('4Kimage.jpg')
        load = load.resize((400,300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        img = Label(self, image =render)
        img.image=render
        img.place(x=0,y=0)
        self.imgPath = '4Kimage.jpg'
        # status = Label(root, text="showimg...", bd =1,relief= SUNKEN, anchor=W)
        # status.pack(side=BOTTOM, fill=X)

    def showImg4(self):
        Frame.__init__(self)
        self.init_window()
        load = Image.open('8Kimage.jpg')
        load = load.resize((400,300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)

        img = Label(self, image =render)
        img.image=render
        img.place(x=0,y=0)

        # status = Label(root, text="showimg...", bd =1,relief= SUNKEN, anchor=W)
        # status.pack(side=BOTTOM, fill=X)

        self.imgPath = '8Kimage.jpg'

    def showText(self):
        text = Label(self, text="woo woo woo..!")
        text.pack()

    def client_exit(self):
        exit()


    def init_window(self):
        self.master.title("GUI")
        self.pack(fill=BOTH, expand=1)

    # quitButton = Button(self, text="Quit", command=self.client_exit)
    # quitButton.place(x=0, y=0)

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Save")
        file.add_command(label="Exit", command=menu.quit)  # self.client_exit())
        menu.add_cascade(label="File", menu=file)

        edit = Menu(menu)
        edit.add_command(label = "720Image", command=self.showImg)
        edit.add_command(label = "1KImage", command=self.showImg2)
        edit.add_command(label = "4KImage", command=self.showImg3)
        edit.add_command(label = "8KImage", command=self.showImg4)
        # edit.add_cascade(label="Show Text", menu=self.showText())

        menu.add_cascade(label="Edit", menu=edit)

        # status = Label(root, text="activate...", bd=1, relief=SUNKEN, anchor=W)
        # status.pack(side=BOTTOM, fill=X)

        #############################################
        btn = Button(self, text='serial',width = '54', command=self.serial_event)
        btn.place(x=400, y=300)

        btn2 = Button(self, text = 'CUDA', width = '54', command = self.cuda_event)
        btn2.place(x = 800, y=300)


        frame1 = Frame(self)
        frame1.pack(fill = X)
        lblName1 = Label(frame1, text = "Serial Processing Time", width = 30)
        #lblName1.place(x=0, y=0)
        lblName1.pack(side=LEFT, padx=10, pady=10)

        global entryName1
        entryName1 = Entry(frame1)
        entryName1.pack(side=LEFT, padx=10, expand=False)

        frame2 = Frame(self)
        frame2.pack(fill = X)
        lblName2 = Label(frame2, text = "Parallel Processing Time", width = 30)
        #lblName2.place(x=0, y=0)
        lblName2.pack(side=LEFT, padx=10, pady=10)

        entryName2 = Entry(frame2)
        entryName2.pack(side=LEFT, padx=10, expand=False)


    def serial_event(self):
        inPath=self.imgPath
        outPath = 'result.jpg'
        totalT0 = time.time()
        blackWhite(inPath, outPath, mode='luminosity', log=1)
        totalT1 = time.time()
        entryName1.insert("Serial Processing Time: ",str(float(totalT1 - totalT0)))

        load = Image.open('result.jpg')
        load = load.resize((400,300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=400, y=0)


    def cuda_event(self):
        inPath=self.imgPath
        outPath = 'result2.jpg'
        CudablackWhite(inPath, outPath, mode='luminosity', log=1)
        load = Image.open('result2.jpg')
        load = load.resize((400,300), Image.ANTIALIAS)
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.place(x=800, y=0)
    

if __name__ == '__main__':
    start_time = time.time()

    root = Tk()
    app = Window(root)
    root,mainloop()

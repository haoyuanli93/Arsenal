import numpy as np
from tkinter import *
from tkinter import ttk
import glob
import argparse

####################################################################################################
# [USER] Specify the run number
####################################################################################################
# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str,
                    help="Specify the input folder containing all the patterns to classify.")

args = parser.parse_args()
input_folder = args.input_folder

######################################################
# Define Variables
######################################################
label = None
image_files = None
image_number = None
image_index = None


######################################################
# Define functions
######################################################
def load_image_fun():
    global pattern, Img, image_index, label
    pattern = PhotoImage(master=mainframe, file=input_folder + '/image_{}.png'.format(image_index)).zoom(3)

    Img.config(image=pattern)
    if not (label is None):
        img_title.set("Diffraction Pattern Index: %d ; Current Label: %s" % (image_index, label[image_index]))
    return


def load_data_fun():
    global image_files, image_number, data_address, image_index
    image_files = np.sort(glob.glob(data_address.get() + '/image_*.png'))
    image_number = len(image_files)
    image_index = 0
    load_image_fun()
    return


def create_label_fun():
    global image_files, image_number, label, create_label
    if test_data():
        label = np.zeros(image_number)
        np.save(create_label.get(), label)
    else:
        return


def load_label_fun():
    global label, load_label
    label = np.load(load_label.get())
    return


def save_label_fun():
    global label, save_label
    np.save(save_label.get(), label)
    print("There are totally {} patterns classified.".format(label.shape[0]) +
          "{} of them are single hits.".format(int(np.sum(label[np.abs(label - 1) <= 0.1]))))
    return


def next_image_fun(*args):
    global image_files, image_index, image_number, pattern, Img

    if image_index <= image_number - 2:
        image_index += 1
        load_image_fun()
        new_label_text.set('0')
    return


def next_run_number_fun():
    pass
    return


def previous_image_fun(*args):
    global image_files, image_index, image_number, pattern, Img
    if image_index >= 1:
        image_index -= 1
        load_image_fun()
        new_label_text.set('0')
    return


def previous_run_number_fun():
    pass
    return


def set_label_fun():
    global label, image_index, new_label_entry
    label[image_index] = int(new_label_entry.get())
    return


def test_data():
    # Use this test function to see whether the data is loaded
    global image_files
    if image_files is None:
        print('data is not loaded')
        return False
    return True


def set_good(*args):
    # 0 indicates that this image was not inspected
    # 1 indicates that this is a good single hit
    # 2 indicates that this is a bad hit
    global new_label_text
    new_label_text.set('1')
    set_label_fun()
    return


def set_bad(*args):
    # 0 indicates that this image was not inspected
    # 1 indicates that this is a good single hit
    # 2 indicates that this is a bad hit
    global new_label_text
    new_label_text.set('2')
    set_label_fun()
    return


######################################################
# Setting up the Whole frame
######################################################

root = Tk()
root.title("Manual Classifier")
root.option_add("*font", "Helvetica 20")

style = ttk.Style()
style.configure('.', font=('Helvetica', 20))

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# Title for the imagine demonstration
img_title = StringVar()
img_title.set('Start the Classifier')
ttk.Label(mainframe, textvariable=img_title).grid(column=0, row=0)
# Image
pattern = PhotoImage(master=mainframe,
                     file='/reg/neh/home/haoyuan/Documents' +
                          '/my_repos/Manual-Classification/asset/Start.png').zoom(3)

Img = Label(mainframe, image=pattern)
Img.grid(column=0, row=1, rowspan=8)

# Load data
data_address = StringVar()
data_address.set('../output/')

Data_address = ttk.Entry(mainframe, textvariable=data_address, width=50)
Data_address.grid(column=2, row=1, columnspan=2)

load_data_button = ttk.Button(mainframe, text='Load Data', command=load_data_fun, width=10)
load_data_button.grid(column=1, row=1)

# Create label 
create_label = StringVar()
create_label.set('../output/label.npy')

Create_label = ttk.Entry(mainframe, textvariable=create_label, width=50)
Create_label.grid(column=2, row=2)

create_label_button = ttk.Button(mainframe, text='Create Label', command=create_label_fun, width=10)
create_label_button.grid(column=1, row=2)

# Load label
load_label = StringVar()
load_label.set('../output/label.npy')

Load_label = ttk.Entry(mainframe, textvariable=load_label, width=50)
Load_label.grid(column=2, row=3)

load_label_button = ttk.Button(mainframe, text='Load Label', command=load_label_fun, width=10)
load_label_button.grid(column=1, row=3)

# Save label
save_label = StringVar()
save_label.set('../output/label.npy')

Save_label = ttk.Entry(mainframe, textvariable=save_label, width=50)
Save_label.grid(column=2, row=4)

save_label_button = ttk.Button(mainframe, text='Save Label', command=save_label_fun, width=10)
save_label_button.grid(column=1, row=4)

# Change run number and image
next_image = ttk.Button(mainframe, text='Next Image', command=next_image_fun, width=18)
next_image.grid(column=1, row=5, sticky='e')

next_run_number = ttk.Button(mainframe, text='Next Run Number', command=next_run_number_fun, width=18)
next_run_number.grid(column=2, row=5, sticky='w')

previous_image = ttk.Button(mainframe, text='Previous Image', command=previous_image_fun, width=18)
previous_image.grid(column=1, row=6, sticky='e')

previous_run_number = ttk.Button(mainframe, text='Previous Run Number',
                                 command=previous_run_number_fun, width=18)
previous_run_number.grid(column=2, row=6, sticky='w')

# Show current run number, image index
new_label_label = ttk.Label(mainframe, text='New Label')
new_label_label.grid(column=1, row=7, sticky='e')

new_label_text = StringVar()
new_label_text.set('0')
new_label_entry = ttk.Entry(mainframe, text=new_label_text, width=6)
new_label_entry.grid(column=2, row=7, sticky='w')

set_label = ttk.Button(mainframe, text='Set Label', command=set_label_fun)
set_label.grid(column=1, row=8, sticky='e')

# Pad to get more space
for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)
# Add a size grip to change the shape easier
ttk.Sizegrip(root).grid(column=999, row=999, sticky=(S, E))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# Add keyboard binding
root.bind('<Right>', next_image_fun)
root.bind('<Left>', previous_image_fun)
root.bind('1', set_good)
root.bind('2', set_bad)
root.focus_set()

root.mainloop()

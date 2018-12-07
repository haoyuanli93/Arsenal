import numpy as np
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import os
import glob
import argparse
import time
import datetime
import subprocess

####################################################################################################
# [AUTO] Define some auxiliary function
####################################################################################################
# Get a time stemp.
"""
Because I do not want to add to much dependence to this script, 
I will add some more functions which can be find in the arsenal package.  
"""


def get_time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


####################################################################################################
# [USER] Specify the run number
####################################################################################################
# Parse the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str,
                    help="Specify the input folder containing all the patterns to classify.")

args = parser.parse_args()
input_folder = os.path.abspath(args.input_folder)

print("Input folder: ", input_folder)
# Define the label file
label_file = input_folder + r'/label.npy'

######################################################
# Define Variables
######################################################
label = None
image_files = None
image_number = None
image_index = None


######################################################
# Define IO and label creation functions
######################################################
def load_image_function():
    """
    Load teh image
    :return:
    """
    global pattern, Img, image_index, label
    pattern = PhotoImage(master=mainframe,
                         file=os.path.abspath(input_folder +
                                              '/image_{}.png'.format(image_index)
                                              )).zoom(3)

    Img.config(image=pattern)
    if not (label is None):
        img_title.set("Diffraction Pattern "
                      "Index: {}; Current Label: {}".format(image_index,
                                                            label[image_index]))
    return


def load_data_function():
    """
    Load the data
    :return:
    """

    global image_files, image_number, input_folder, image_index

    # Get the list of all pattern files
    image_files = np.sort(glob.glob(input_folder + '/image_*.png'))

    # Find out how many patterns are there in this folder
    image_number = len(image_files)

    # Show how many patterns are there in this folder
    messagebox.showinfo(title="Information",
                        message="There are totally "
                                "{} patterns in this folder.".format(image_number))

    # Starting from the first pattern.
    image_index = 0

    # Load the first pattern.
    load_image_function()
    return


def create_label_function():
    """
    Create the label file
    :return:
    """
    global image_files, image_number, label_file, input_folder, label

    # Check if the data has been loaded.
    if not (image_files is None):
        # Check if there is any existing label file.
        if os.path.isfile(label_file):

            # Get a time stamp and the backup name
            time_stamp = get_time_stamp()
            backup_name = input_folder + "/label_backup_{}.npy".format(time_stamp)

            # Backup the existing label file and create a new file
            subprocess.call(["mv", label_file, backup_name])

            # Send out the message
            messagebox.showinfo(title="Information",
                                message=" The a label.npy file" +
                                        " already exists in the directory \n \n" +
                                        "{}\n \n".format(input_folder) +
                                        "Therefore, the original label " +
                                        "file is now renamed to be\n \n" +
                                        "{}\n \n".format(backup_name) +
                                        "A new label file is create at \n \n" +
                                        "{}".format(label_file))

            label = np.zeros(image_number, dtype=np.int64)
            np.save(label_file, label)
        else:
            # Send out the message
            messagebox.showinfo(title="Information",
                                message="A label file is create at \n" +
                                        "{}".format(label_file))
            label = np.zeros(image_number, dtype=np.int64)
            np.save(label_file, label)
    else:
        messagebox.showinfo(title="Information",
                            message="Please load the data first before creating the label."
                                    "To load the data, please click the \"Load Data\" button.")


def load_label_function():
    """Load the label."""
    global label, label_file

    # Check if the label file exist:
    if not (os.path.isfile(label_file)):
        messagebox.showinfo(title="Information",
                            message="The label file does not exist. "
                                    "Please create it before loading it."
                                    "To create the label file, "
                                    "please click the \'Load Label\' button. ")
        return

    else:
        label = np.load(label_file)


def save_label_function():
    """
    Save the label.
    :return:
    """
    global label, label_file, image_number
    np.save(label_file, label)
    messagebox.showinfo(title="Information",
                        message="Save the label to the file\n \n " +
                                "{}\n \n".format(label_file) +
                                "There are totally {} patterns.".format(image_number))

    print("There are totally {} patterns classified.".format(image_number))

    unique_label = np.unique(label)
    for value in unique_label:
        print("{} patterns are classified as {}.".format(np.count_nonzero(label == value), value))


def next_image_function(*key):
    # Nothing
    if key is None:
        pass

    global image_files, image_index, image_number, pattern, Img

    if image_index <= image_number - 2:
        image_index += 1
        load_image_function()
        new_label_text.set('0')
    return


def previous_image_function(*key):
    # Nothing
    if key is None:
        pass

    global image_files, image_index, image_number, pattern, Img
    if image_index >= 1:
        image_index -= 1
        load_image_function()
        new_label_text.set('0')
    return


###################################################################################################
# Specifying the label
###################################################################################################

def set_label_function():
    global label, image_index, new_label_entry
    label[image_index] = int(new_label_entry.get())
    return


def set_one(*key):
    if key is None:
        pass

    # 0 indicates that this image was not inspected
    # 1 indicates that this is a good single hit
    # 2 indicates that this is a bad hit
    global new_label_text
    new_label_text.set('1')
    set_label_function()
    return


def set_two(*key):
    if key is None:
        pass

    # 0 indicates that this image was not inspected
    # 1 indicates that this is a good single hit
    # 2 indicates that this is a bad hit
    global new_label_text
    new_label_text.set('2')
    set_label_function()
    return


def set_three(*key):
    if key is None:
        pass

    # 0 indicates that this image was not inspected
    # 1 indicates that this is a good single hit
    # 2 indicates that this is a bad hit
    global new_label_text
    new_label_text.set('3')
    set_label_function()
    return


def set_four(*key):
    if key is None:
        pass

    # 0 indicates that this image was not inspected
    # 1 indicates that this is a good single hit
    # 2 indicates that this is a bad hit
    global new_label_text
    new_label_text.set('4')
    set_label_function()
    return


######################################################
# Set up the framework
######################################################

root = Tk()
root.title("Manual Classifier")
root.option_add("*font", "Times 18")

style = ttk.Style()
style.map("TEntry",
          fieldbackground=[("active", "white"),
                           ("disabled", "white")])
style.configure('.', font=('Times', 18))

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# Title for the imagine demonstration
img_title = StringVar()
img_title.set('Start the Classifier')
ttk.Label(mainframe, textvariable=img_title).grid(column=0, row=0)

# Show the label meaning
label_meaning = StringVar()
label_meaning.set('1=Good  2=Bad  3=Water  4=Other')
ttk.Label(mainframe, textvariable=label_meaning).grid(column=2, row=0, columnspan=2)

######################################################
# Specify entries in the framework
######################################################
# Image
pattern = PhotoImage(master=mainframe,
                     file='/reg/neh/home/haoyuan/Documents' +
                          '/my_repos/Manual-Classification/asset/Start.png').zoom(3)

Img = Label(mainframe, image=pattern)
Img.grid(column=0, row=1, rowspan=6)

# Load data
data_address = StringVar()
data_address.set(input_folder[-40:])

Data_address = ttk.Entry(mainframe,
                         textvariable=data_address,
                         width=40,
                         state=DISABLED)
Data_address.grid(column=2, row=1, columnspan=2)

load_data_button = ttk.Button(mainframe, text='Load Data', command=load_data_function, width=18)
load_data_button.grid(column=1, row=1)

# Create label 
create_label = StringVar()
create_label.set((input_folder + '/label.npy')[-40:])

Create_label = ttk.Entry(mainframe,
                         textvariable=create_label,
                         width=40,
                         state=DISABLED)
Create_label.grid(column=2, row=2, columnspan=2)

create_label_button = ttk.Button(mainframe,
                                 text='Create Label',
                                 command=create_label_function,
                                 width=18)
create_label_button.grid(column=1, row=2)

# Load label
load_label = StringVar()
load_label.set((input_folder + '/label.npy')[-40:])

Load_label = ttk.Entry(mainframe,
                       textvariable=load_label,
                       width=40,
                       state=DISABLED)
Load_label.grid(column=2, row=3, columnspan=2)

load_label_button = ttk.Button(mainframe,
                               text='Load Label',
                               command=load_label_function,
                               width=18)
load_label_button.grid(column=1, row=3)

# Save label
save_label = StringVar()
save_label.set((input_folder + '/label.npy')[-40:])

Save_label = ttk.Entry(mainframe,
                       textvariable=save_label,
                       width=40,
                       state=DISABLED)
Save_label.grid(column=2, row=4, columnspan=2)

save_label_button = ttk.Button(mainframe,
                               text='Save Label',
                               command=save_label_function,
                               width=18)
save_label_button.grid(column=1, row=4)

# Change image
next_image = ttk.Button(mainframe,
                        text='Next Image',
                        command=next_image_function,
                        width=18)
next_image.grid(column=1, row=5, sticky='e')

previous_image = ttk.Button(mainframe,
                            text='Previous Image',
                            command=previous_image_function,
                            width=18)
previous_image.grid(column=1, row=6, sticky='e')

# Show current run number, image index
new_label_label = ttk.Label(mainframe, text='New Label')
new_label_label.grid(column=2, row=5, sticky='e')

new_label_text = StringVar()
new_label_text.set('0')
new_label_entry = ttk.Entry(mainframe, text=new_label_text, width=6)
new_label_entry.grid(column=3, row=5, sticky='w')

set_label = ttk.Button(mainframe, text='Set Label', command=set_label_function)
set_label.grid(column=2, row=6, sticky='e')

# Pad to get more space
for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

# Add a size grip to change the shape easier
ttk.Sizegrip(root).grid(column=999, row=999, sticky=(S, E))
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

# Add keyboard binding
root.bind('<Right>', next_image_function)
root.bind('<Left>', previous_image_function)
root.bind('1', set_one)
root.bind('2', set_two)
root.bind('3', set_three)
root.bind('4', set_four)
root.focus_set()

root.mainloop()

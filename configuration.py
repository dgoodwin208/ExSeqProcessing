from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import subprocess




def browse_button(tag):
    global folder_path
    filename = filedialog.askdirectory()
    folder_path[tag]=filename
    

global folder_path
folder_path = {}

try:
	root = Tk()

	basename = StringVar()
	ch_prefix = StringVar()
	fmt = StringVar()
	accel = StringVar()
	total_rounds = IntVar()
	ref_round = IntVar()

	root.title("ExSeqProcessing Configuration")

	Label(root, text="Input", font="Tahoma 10 bold").grid(row=0, column=0, padx=10, pady=1,  sticky=W)
	Label(root, text="Path:", font="Tahoma 10 bold").grid(row=0, column=1, padx=10, pady=1, sticky=W)
	B1 = Button(text="Browse", command= lambda: browse_button("input")).grid(row=0, column=2, padx=10, pady=1, sticky=E)

	Label(root, text="Basename:", font="Tahoma 10 bold").grid(row=1, column=1, padx=10, pady=1,  sticky=W)
	Entry(root, textvariable=basename).grid(row=1, column=2, sticky=E, pady=1)

	Label(root, text="Channel Prefix:", font="Tahoma 10 bold").grid(row=2, column=1, padx=10, pady=1,  sticky=W)
	Entry(root, textvariable=ch_prefix).grid(row=2, column=2, sticky=E, pady=1)

	sep = ttk.Separator(root, orient=HORIZONTAL)
	sep.grid(row=3, column=0, columnspan=4, sticky="ew")

	Label(root, text="Output", font="Tahoma 10 bold").grid(row=4, column=0, padx=10, pady=1,  sticky=W)
	Label(root, text="Path:", font="Tahoma 10 bold").grid(row=4, column=1, padx=10, pady=1, sticky=W)
	B1 = Button(text="Browse", command= lambda: browse_button("output")).grid(row=4, column=2, padx=10, pady=1, sticky=E)

	Label(root, text="Format:", font="Tahoma 10 bold").grid(row=5, column=1, padx=10, pady=1,  sticky=W)
	Radiobutton(root, text="tiff", variable=fmt, value="tiff").grid(row=5, column=2, padx=10, pady=1,  sticky=W)

	Radiobutton(root, text="hdf5", variable=fmt, value="hdf5").grid(row=6, column=2, padx=10, pady=1,  sticky=W)

	sep = ttk.Separator(root, orient=HORIZONTAL)
	sep.grid(row=7, column=0, columnspan=4, sticky="ew")

	Label(root, text="Log Directory", font="Tahoma 10 bold").grid(row=8, column=0, padx=10, pady=1,  sticky=W)
	Label(root, text="Path:", font="Tahoma 10 bold").grid(row=8, column=1, padx=10, pady=1, sticky=W)
	B1 = Button(text="Browse", command= lambda: browse_button("log")).grid(row=8, column=2, padx=10, pady=1, sticky=E)

	sep = ttk.Separator(root, orient=HORIZONTAL)
	sep.grid(row=9, column=0, columnspan=4, sticky="ew")

	Label(root, text="Temporary Storage", font="Tahoma 10 bold").grid(row=10, column=0, padx=10, pady=1,  sticky=W)
	Label(root, text="Path:", font="Tahoma 10 bold").grid(row=10, column=1, padx=10, pady=1, sticky=W)
	B1 = Button(text="Browse", command= lambda: browse_button("temp")).grid(row=10, column=2, padx=10, pady=1, sticky=E)

	sep = ttk.Separator(root, orient=HORIZONTAL)
	sep.grid(row=11, column=0, columnspan=4, sticky="ew")

	Label(root, text="Round Information", font="Tahoma 10 bold").grid(row=12, column=0, padx=10, pady=1,  sticky=W)
	Label(root, text="Total Rounds:", font="Tahoma 10 bold").grid(row=12, column=1, padx=10, pady=1,  sticky=W)
	Entry(root, textvariable=total_rounds).grid(row=12, column=2, sticky=E, pady=1)

	Label(root, text="Reference Round:", font="Tahoma 10 bold").grid(row=13, column=1, padx=10, pady=1,  sticky=W)
	Entry(root, textvariable=ref_round).grid(row=13, column=2, sticky=E, pady=1)

	sep = ttk.Separator(root, orient=HORIZONTAL)
	sep.grid(row=14, column=0, columnspan=4, sticky="ew")

	Label(root, text="Acceleration", font="Tahoma 10 bold").grid(row=15, column=0, padx=10, pady=1,  sticky=W)
	Radiobutton(root, text="GPU", variable=accel, value="gpu").grid(row=15, column=2, padx=10, pady=1,  sticky=W)

	Radiobutton(root, text="CPU", variable=accel, value="cpu").grid(row=16, column=2, padx=10, pady=1,  sticky=W)

	sep = ttk.Separator(root, orient=HORIZONTAL)
	sep.grid(row=17, column=0, columnspan=4, sticky="ew")

	B1 = Button(text="Submit", command= lambda: root.destroy(), font="Tahoma 11 bold").grid(row=18, column=1, padx=10, pady=10, sticky=E)

	root.mainloop()

	basename = basename.get()
	ch_prefix = ch_prefix.get()
	fmt = fmt.get()
	accel = accel.get()
	total_rounds = total_rounds.get()
	ref_round = ref_round.get()
except:
	print("No display connected, fill the below details to form configuration file: ")

	path=subprocess.check_output('read -e -p "Enter path for input files: " var ; echo $var',shell=True).rstrip()
	folder_path["input"] = path.decode('utf-8')
	basename = input("Enter the basename: ")
	ch_prefix = input("Enter the channel prefix: ")
	path=subprocess.check_output('read -e -p "Enter path for output files: " var ; echo $var',shell=True).rstrip()
	folder_path["output"] = path.decode('utf-8')	
	fmt = input("Enter 1 for tiff, Enter 2 for hdf5: ")
	path=subprocess.check_output('read -e -p "Enter path for temporary storage: " var ; echo $var',shell=True).rstrip()
	folder_path["temp"] = path.decode('utf-8')	
	path=subprocess.check_output('read -e -p "Enter path for log files: " var ; echo $var',shell=True).rstrip()
	folder_path["log"] = path.decode('utf-8')	
	total_rounds = input("Enter the total number of rounds: ")
	ref_round = input("Enter the reference round number: ")
	accel = input("Enter 1 for GPU acceleration, Enter 2 for CPU acceleration: ")

	fmt=int(fmt)
	if(fmt == 1):
		fmt = "tiff"
	else:
		fmt = "hdf5"

	total_rounds = int(total_rounds)
	ref_round = int(ref_round)

	accel=int(accel)
	if(accel == 1):
		accel = "gpu"
	else:
		accel = "cpu"

print("Input path:", folder_path["input"])
print("basename:", basename)
print("channel prefix:", ch_prefix)
print("output path:", folder_path["output"])
print("format:", fmt)
print("tmp:", folder_path["temp"])
print("Log:", folder_path["log"])
print("Total Rounds:", total_rounds)
print("Reference Round:", ref_round)
print("Acceleration:", accel)

channels = "'" + ch_prefix + "00','" + ch_prefix + "01SHIFT','" + ch_prefix + "02SHIFT','" + ch_prefix + "03SHIFT'"

f=open("configuration.cfg","w+")

f.write("input_path="+folder_path["input"])
f.write("\n")
f.write("basename="+str(basename))
f.write("\n")
f.write("channels="+str(channels))
f.write("\n")
f.write("output_path="+folder_path["output"])
f.write("\n")
f.write("format="+str(fmt))
f.write("\n")
f.write("log_path="+folder_path["log"])
f.write("\n")
f.write("tmp_path="+folder_path["temp"])
f.write("\n")
f.write("total_rounds="+str(total_rounds))
f.write("\n")
f.write("reference_round="+str(ref_round))
f.write("\n")
f.write("acceleration="+str(accel))
f.write("\n")
f.close()

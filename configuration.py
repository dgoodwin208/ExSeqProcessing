from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import subprocess
import re




def browse_button(tag):
    global folder_path
    filename = filedialog.askdirectory()
    folder_path[tag]=filename

def read_matlab_parameters():
    try:
        f = open("loadParameters.m")
    except FileNotFoundError:
        f = open("loadParameters.m.template")

    comp_keyvalue = re.compile("^([^% ]*) *= *([^;]*);.*")
    config_map = {}
    for line in f:
        m = comp_keyvalue.search(line)
        if m != None:
            key = m.group(1)
            val = m.group(2).replace("'", "").lstrip("{").rstrip("}")
            config_map[key] = val

    return config_map


global folder_path
folder_path = {"input":"", "output":"", "temp":"", "log":""}

config_map = read_matlab_parameters()
print(config_map)

try:
	root = Tk()

	basename = StringVar(root, value=config_map.get("params.FILE_BASENAME"))
	channels = StringVar(root, value=config_map.get("params.CHAN_STRS"))
	fmt = StringVar()
	accel = StringVar()
	total_rounds = IntVar(root, value=int(config_map.get("params.NUM_ROUNDS")))
	ref_round = IntVar(root, value=int(config_map.get("params.REFERENCE_ROUND_WARP")))

	root.title("ExSeqProcessing Configuration")

	Label(root, text="Input", font="Tahoma 10 bold").grid(row=0, column=0, padx=10, pady=1,  sticky=W)
	Label(root, text="Path:", font="Tahoma 10 bold").grid(row=0, column=1, padx=10, pady=1, sticky=W)
	B1 = Button(text="Browse", command= lambda: browse_button("input")).grid(row=0, column=2, padx=10, pady=1, sticky=E)

	Label(root, text="Basename:", font="Tahoma 10 bold").grid(row=1, column=1, padx=10, pady=1,  sticky=W)
	Entry(root, textvariable=basename).grid(row=1, column=2, sticky=E, pady=1)

	Label(root, text="Channels:", font="Tahoma 10 bold").grid(row=2, column=1, padx=10, pady=1,  sticky=W)
	Entry(root, textvariable=channels).grid(row=2, column=2, sticky=E, pady=1)

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
	Radiobutton(root, text="GPU_CUDA", variable=accel, value="gpu_cuda").grid(row=15, column=2, padx=10, pady=1,  sticky=W)

	Radiobutton(root, text="CPU", variable=accel, value="cpu").grid(row=16, column=2, padx=10, pady=1,  sticky=W)

	sep = ttk.Separator(root, orient=HORIZONTAL)
	sep.grid(row=17, column=0, columnspan=4, sticky="ew")

	B1 = Button(text="Submit", command= lambda: root.destroy(), font="Tahoma 11 bold").grid(row=18, column=1, padx=10, pady=10, sticky=E)

	root.mainloop()

	basename = basename.get()
	channels = channels.get()
	fmt = fmt.get()
	accel = accel.get()
	total_rounds = total_rounds.get()
	ref_round = ref_round.get()


except:
	print("No display connected, fill the below details to form configuration file: ")

	basename = config_map.get("params.FILE_BASENAME")
	channels = config_map.get("params.CHAN_STRS")
	fmt = ("tiff" if config_map.get("params.IMAGE_EXT") == "tif" else "hdf5")
	temp = config_map.get("params.tempDir")
	log = config_map.get("params.logDir")
	total_rounds = int(config_map.get("params.NUM_ROUNDS"))
	ref_round = int(config_map.get("params.REFERENCE_ROUND_WARP"))
	accel = ("gpu_cuda" if config_map.get("params.USE_GPU_CUDA") == "true" else "cpu")

	path=subprocess.check_output('read -e -p "Enter path for input files: " var ; echo $var',shell=True).rstrip()
	folder_path["input"] = path.decode('utf-8')
	ret = input("Enter the basename (default: %s): " % basename)
	if ret != "":
		basename = ret
	ret = input("Enter the channels (default: %s): " % channels)
	if ret != "":
		channels = ret
	path=subprocess.check_output('read -e -p "Enter path for output files: " var ; echo $var',shell=True).rstrip()
	folder_path["output"] = path.decode('utf-8')
	ret = input("Enter 1 for tiff, Enter 2 for hdf5 (default: %s): " % fmt)
	if ret != "":
		fmt_id = int(ret)
		if(fmt_id == 1):
			fmt = "tiff"
		else:
			fmt = "hdf5"
	path=subprocess.check_output('read -e -p "Enter path for temporary storage (default: %s): " var ; echo $var' % temp,shell=True).rstrip().decode('utf-8')
	if path != "":
		folder_path["temp"] = path
	else:
		folder_path["temp"] = temp
	path=subprocess.check_output('read -e -p "Enter path for log files (default: %s): " var ; echo $var' % log,shell=True).rstrip().decode('utf-8')
	if path != "":
		folder_path["log"] = path
	else:
		folder_path["log"] = log
	ret = input("Enter the total number of rounds (default: %d): " % total_rounds)
	if ret != "":
		total_rounds = int(ret)
	ret = input("Enter the reference round number (default: %d): " % ref_round)
	if ret != "":
		ref_round = int(ret)
	ret = input("Enter 1 for GPU-CUDA acceleration, Enter 2 for CPU acceleration (default: %s): " % accel)
	if ret != "":
		accel_id = int(ret)
		if(accel_id == 1):
			accel = "gpu_cuda"
		else:
			accel = "cpu"

print("Input path:", folder_path["input"])
print("basename:", basename)
print("channels:", channels)
print("output path:", folder_path["output"])
print("format:", fmt)
print("tmp:", folder_path["temp"])
print("Log:", folder_path["log"])
print("Total Rounds:", total_rounds)
print("Reference Round:", ref_round)
print("Acceleration:", accel)

channels = ','.join([ "'"+x+"'" for x in channels.split(',') ])

f=open("configuration.cfg","w+")

f.write("input_path="+folder_path["input"])
f.write("\n")
f.write("basename="+basename)
f.write("\n")
f.write("channels="+channels)
f.write("\n")
f.write("output_path="+folder_path["output"])
f.write("\n")
f.write("format="+fmt)
f.write("\n")
f.write("log_path="+folder_path["log"])
f.write("\n")
f.write("tmp_path="+folder_path["temp"])
f.write("\n")
f.write("total_rounds="+str(total_rounds))
f.write("\n")
f.write("reference_round="+str(ref_round))
f.write("\n")
f.write("acceleration="+accel)
f.write("\n")
f.close()

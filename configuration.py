import subprocess
import re
import sys
import shutil


try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

if sys.version_info.major != 2 and sys.version_info.major != 3:
    raise NotImplementedError("not suppport Python version: %d" % sys.version_info.major);


def input_param(message):
    if sys.version_info.major == 2:
        ret = raw_input(message)
    elif sys.version_info.major == 3:
        ret = input(message)

    return ret;


def browse_button(var):
    filename = filedialog.askdirectory()
    var.set(filename)

def read_matlab_parameters():

    f = open("loadParameters.m")

    comp_keyvalue = re.compile("^([^% ]*) *= *([^;]*);.*")
    config_map = {}
    config_linenum = {}
    config_lines = []
    count = 0
    for line in f:
        m = comp_keyvalue.search(line)
        if m != None:
            key = m.group(1)
            val = m.group(2).replace("'", "").lstrip("{").rstrip("}")
            config_map[key] = val
            config_linenum[key] = count

        config_lines.append(line.rstrip())
        count += 1

    return config_map, config_linenum, config_lines


folder_path = {"input":"", "output":"", "temp":"", "log":""}

config_map, config_linenum, config_lines = read_matlab_parameters()
if "params.logDir" not in config_map:
    config_map["params.logDir"] = "logs"
if "params.tempDir" not in config_map:
    config_map["params.tempDir"] = "tmp"
if "params.INPUT_FILE_PATH" not in config_map:
    print(1)
    config_map["params.INPUT_FILE_PATH"] = ""
    config_linenum["params.INPUT_FILE_PATH"] = len(config_lines)
    config_lines.append("")
if "params.deconvolutionImagesDir" not in config_map:
    config_map["OUTPUT_PATH"] = os.getcwd()
else:
    comp_path = re.compile("(.*)/[^/]*$")
    m = comp_path.search(config_map["params.deconvolutionImagesDir"])
    if m != None:
        config_map["OUTPUT_PATH"] = m.group(1)
    else:
        config_map["OUTPUT_PATH"] = os.getcwd()

#print(config_map)
#print(config_linenum)
#print(config_lines)

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter import filedialog
    root = Tk()

    input_file_path = StringVar(root, config_map.get("params.INPUT_FILE_PATH"))
    output_path = StringVar(root, config_map.get("OUTPUT_PATH"))
    temp = StringVar(root, config_map.get("params.tempDir"))
    log = StringVar(root, config_map.get("params.logDir"))
    basename = StringVar(root, value=config_map.get("params.FILE_BASENAME"))
    channels = StringVar(root, value=config_map.get("params.CHAN_STRS"))
    fmt = StringVar(root, value=("tiff" if config_map.get("params.IMAGE_EXT") == "tif" else "hdf5"))
    accel = StringVar(root, value=("gpu_cuda" if config_map.get("params.USE_GPU_CUDA") == "true" else "cpu"))
    total_rounds = IntVar(root, value=int(config_map.get("params.NUM_ROUNDS")))
    ref_round = IntVar(root, value=int(config_map.get("params.REFERENCE_ROUND_WARP")))

    root.title("ExSeqProcessing Configuration")

    path_entry_width = 60
    row_idx = 0
    Label(root, text="Input", font="Tahoma 10 bold").grid(row=row_idx, column=0, columnspan=2, padx=10, pady=1,  sticky=W)
    row_idx += 1
    Label(root, text="     ", font="Tahoma 10 bold").grid(row=row_idx, column=0,  sticky=W)
    Label(root, text="Path:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1, sticky=W)
    Entry(root, textvariable=input_file_path, width=path_entry_width).grid(row=row_idx, column=2, columnspan=2, pady=1)
    B1 = Button(text="Browse...", command= lambda: browse_button(input_file_path)).grid(row=row_idx, column=4, padx=10, pady=1, sticky=E)

    row_idx += 1
    Label(root, text="Basename:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1,  sticky=W)
    Entry(root, textvariable=basename).grid(row=row_idx, column=2, sticky=W, pady=1)

    row_idx += 1
    Label(root, text="Channels:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1,  sticky=W)
    Entry(root, textvariable=channels).grid(row=row_idx, column=2, sticky=W, pady=1)

    sep = ttk.Separator(root, orient=HORIZONTAL)
    row_idx += 1
    sep.grid(row=row_idx, column=0, columnspan=5, sticky="ew")

    row_idx += 1
    Label(root, text="Output", font="Tahoma 10 bold").grid(row=row_idx, column=0, columnspan=2, padx=10, pady=1,  sticky=W)
    row_idx += 1
    Label(root, text="Path:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1, sticky=W)
    Entry(root, textvariable=output_path, width=path_entry_width).grid(row=row_idx, column=2, columnspan=2, pady=1)
    B1 = Button(text="Browse...", command= lambda: browse_button(output_path)).grid(row=row_idx, column=4, padx=10, pady=1, sticky=E)

    row_idx += 1
    Label(root, text="Format:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1,  sticky=W)
    Radiobutton(root, text="tiff", variable=fmt, value="tiff").grid(row=row_idx, column=2, padx=10, pady=1,  sticky=W)

    row_idx += 1
    Radiobutton(root, text="hdf5", variable=fmt, value="hdf5").grid(row=row_idx, column=2, padx=10, pady=1,  sticky=W)

    sep = ttk.Separator(root, orient=HORIZONTAL)
    row_idx += 1
    sep.grid(row=row_idx, column=0, columnspan=5, sticky="ew")

    row_idx += 1
    Label(root, text="Log Directory", font="Tahoma 10 bold").grid(row=row_idx, column=0, columnspan=2, padx=10, pady=1,  sticky=W)
    row_idx += 1
    Label(root, text="Path:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1, sticky=W)
    Entry(root, textvariable=log, width=path_entry_width).grid(row=row_idx, column=2, columnspan=2, pady=1)
    B1 = Button(text="Browse...", command= lambda: browse_button(log)).grid(row=row_idx, column=4, padx=10, pady=1, sticky=E)

    sep = ttk.Separator(root, orient=HORIZONTAL)
    row_idx += 1
    sep.grid(row=row_idx, column=0, columnspan=5, sticky="ew")

    row_idx += 1
    Label(root, text="Temporary Storage", font="Tahoma 10 bold").grid(row=row_idx, column=0, columnspan=2, padx=10, pady=1,  sticky=W)
    row_idx += 1
    Label(root, text="Path:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1, sticky=W)
    Entry(root, textvariable=temp, width=path_entry_width).grid(row=row_idx, column=2, columnspan=2, pady=1)
    B1 = Button(text="Browse...", command= lambda: browse_button(temp)).grid(row=row_idx, column=4, padx=10, pady=1, sticky=E)

    sep = ttk.Separator(root, orient=HORIZONTAL)
    row_idx += 1
    sep.grid(row=row_idx, column=0, columnspan=5, sticky="ew")

    row_idx += 1
    Label(root, text="Round Information", font="Tahoma 10 bold").grid(row=row_idx, column=0, columnspan=2, padx=10, pady=1,  sticky=W)
    row_idx += 1
    Label(root, text="Total Rounds:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1,  sticky=W)
    Entry(root, textvariable=total_rounds).grid(row=row_idx, column=2, sticky=W, pady=1)

    row_idx += 1
    Label(root, text="Reference Round:", font="Tahoma 10 bold").grid(row=row_idx, column=1, padx=10, pady=1,  sticky=W)
    Entry(root, textvariable=ref_round).grid(row=row_idx, column=2, sticky=W, pady=1)

    sep = ttk.Separator(root, orient=HORIZONTAL)
    row_idx += 1
    sep.grid(row=row_idx, column=0, columnspan=5, sticky="ew")

    row_idx += 1
    Label(root, text="Acceleration", font="Tahoma 10 bold").grid(row=row_idx, column=0, columnspan=2, padx=10, pady=1,  sticky=W)
    row_idx += 1
    Radiobutton(root, text="GPU_CUDA", variable=accel, value="gpu_cuda").grid(row=row_idx, column=2, padx=10, pady=1,  sticky=W)

    row_idx += 1
    Radiobutton(root, text="CPU", variable=accel, value="cpu").grid(row=row_idx, column=2, padx=10, pady=1,  sticky=W)

    sep = ttk.Separator(root, orient=HORIZONTAL)
    row_idx += 1
    sep.grid(row=row_idx, column=0, columnspan=5, sticky="ew")

    row_idx += 1
    B1 = Button(text="Submit", command= lambda: root.destroy(), font="Tahoma 11 bold").grid(row=row_idx, column=1, columnspan=5, padx=10, pady=10)

    root.mainloop()

    folder_path["input"] = input_file_path.get()
    folder_path["output"] = output_path.get()
    folder_path["log"] = log.get()
    folder_path["temp"] = temp.get()

    basename = basename.get()
    channels = channels.get()
    fmt = fmt.get()
    accel = accel.get()
    total_rounds = total_rounds.get()
    ref_round = ref_round.get()


except:
    print("No display connected, fill the below details to form configuration file: ")

    input_file_path = config_map.get("params.INPUT_FILE_PATH")
    output_path = config_map.get("OUTPUT_PATH")
    basename = config_map.get("params.FILE_BASENAME")
    channels = config_map.get("params.CHAN_STRS")
    fmt = ("tiff" if config_map.get("params.IMAGE_EXT") == "tif" else "hdf5")
    temp = config_map.get("params.tempDir")
    log = config_map.get("params.logDir")
    total_rounds = int(config_map.get("params.NUM_ROUNDS"))
    ref_round = int(config_map.get("params.REFERENCE_ROUND_WARP"))
    accel = ("gpu_cuda" if config_map.get("params.USE_GPU_CUDA") == "true" else "cpu")

    path=subprocess.check_output('read -e -p "Enter path for input files (default: %s): " var ; echo $var' % input_file_path,shell=True).rstrip().decode('utf-8')
    if path != "":
        folder_path["input"] = path
    else:
        folder_path["input"] = input_file_path
    ret = input_param("Enter the basename (default: %s): " % basename)
    if ret != "":
        basename = ret
    ret = input_param("Enter the channels (default: %s): " % channels)
    if ret != "":
        channels = ret
    path=subprocess.check_output('read -e -p "Enter path for output files (default: %s): " var ; echo $var' % output_path,shell=True).rstrip().decode('utf-8')
    if path != "":
        folder_path["output"] = path
    else:
        folder_path["output"] = output_path
    ret = input_param("Enter 1 for tiff, Enter 2 for hdf5 (default: %s): " % fmt)
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
    ret = input_param("Enter the total number of rounds (default: %d): " % total_rounds)
    if ret != "":
        total_rounds = int(ret)
    ret = input_param("Enter the reference round number (default: %d): " % ref_round)
    if ret != "":
        ref_round = int(ret)
    ret = input_param("Enter 1 for GPU-CUDA acceleration, Enter 2 for CPU acceleration (default: %s): " % accel)
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


config_lines[config_linenum["params.INPUT_FILE_PATH"]] = "params.INPUT_FILE_PATH = '%s';" % folder_path["input"]
config_lines[config_linenum["params.logDir"]] = "params.logDir = '%s';" % folder_path["log"]
config_lines[config_linenum["params.tempDir"]] = "params.tempDir = '%s';" % folder_path["temp"]
config_lines[config_linenum["params.deconvolutionImagesDir"]] = "params.deconvolutionImagesDir = '%s/1_deconvolution';" % folder_path["output"]
config_lines[config_linenum["params.colorCorrectionImagesDir"]] = "params.colorCorrectionImagesDir = '%s/2_color-correction';" % folder_path["output"]
config_lines[config_linenum["params.normalizedImagesDir"]] = "params.normalizedImagesDir = '%s/3_normalization';" % folder_path["output"]
config_lines[config_linenum["params.registeredImagesDir"]] = "params.registeredImagesDir = '%s/4_registration';" % folder_path["output"]
config_lines[config_linenum["params.punctaSubvolumeDir"]] = "params.punctaSubvolumeDir = '%s/5_puncta-extraction';" % folder_path["output"]
config_lines[config_linenum["params.basecallingResultsDir"]] = "params.basecallingResultsDir = '%s/6_base-calling';" % folder_path["output"]
config_lines[config_linenum["params.FILE_BASENAME"]] = "params.FILE_BASENAME = '%s';" % basename
config_lines[config_linenum["params.CHAN_STRS"]] = "params.CHAN_STRS = {%s};" % channels
config_lines[config_linenum["params.IMAGE_EXT"]] = "params.IMAGE_EXT = '%s';" % ("tif" if fmt == "tiff" else "h5")
config_lines[config_linenum["params.USE_GPU_CUDA"]] = "params.USE_GPU_CUDA = %s;" % ("true" if accel == "gpu_cuda" else "false")
config_lines[config_linenum["params.NUM_ROUNDS"]] = "params.NUM_ROUNDS = %d;" % total_rounds
config_lines[config_linenum["params.REFERENCE_ROUND_WARP"]] = "params.REFERENCE_ROUND_WARP = %d;" % ref_round
config_lines[config_linenum["params.REFERENCE_ROUND_PUNCTA"]] = "params.REFERENCE_ROUND_PUNCTA = %d;" % ref_round

#print(config_lines)

try:
    shutil.copyfile('loadParameters.m', 'loadParameters.m.bak')

    with open("loadParameters.m","w") as f:
        f.writelines("\n".join(config_lines))
except:
    print("cannot write out loadParameters.m")


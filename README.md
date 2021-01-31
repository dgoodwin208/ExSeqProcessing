# ExSeqProcessing

This is the software library for processing Expansion Sequencing, ExSeq, experiments as seen in [Alon et al 2021](https://doi.org/10.1126/science.aax2656). This pipeline takes multi-round in situ sequencing data (including morphological information for spatial context) and convert it into RNA reads in 3D. This software has succesfully processed over 300 fields of view of ExSeq data, corresponding to dozens of terabytes of data, and has helped illucidate biological phenomena at the nanoscale in neuroscience and between cell types in metastatic breast cancer. 

# Getting Started

For the fastest path to exploring the steps of the ExSeqProcessing pipeline, we included a script which will simulate ExSeq data and then process that data. You can find that script in `analysis/simulator/runSimulatorRound.m`. That script should take about 10 minutes to run end-to-end on an modern laptop and should be a good introduction to the file format structure. 

Please refer to the [Wiki](https://github.com/dgoodwin208/ExSeqProcessing/wiki) for information and a tutorial, complete with a sample ExSeq dataset. 

We are grateful for all requests and questions - there will surely be bugs and we want to fix them. It is important to us that this software is a useful contribution to the community!

# Overview
In order to use this pipeline, your data must be formatted in way that can be ingested. You must create grayscale 3D images (either tiff or hdf5) and place them into a folder structure [explained here](https://github.com/dgoodwin208/ExSeqProcessing/wiki/Input-Data-Preparation). 

# Under development
For larger experiments of many fields of view tiling a complete biological sample, we have been developing tools to assist with the automation and handling of challenges. We call this BigEXP and it specifically aims to help with the registration step of large samples. In the ideal case, the experimentalist can physically align the sample so that each field of view can be processed independently. However, in the case of large samples that need to be physically handled often, the assumption that each field of view will be aligned to itself across the sequencing rounds does not hold. BigEXP can used to register all the samples in the situation that fields of view can not be processed independently. This is still in a rough stage, and teams that would like to use this feature can post an issue or email dgoodwin at mit.  

# Acknowledgements
This software pipeline has been a successful multi-year, multi-team collaboration. Specifically, the Boyden Lab would like to express gratitude to Dr. Yosuke Bando from [Kioxia](https://www.kioxia.com/en-us/top.html) (formerly Toshiba Memory) for his leadership in building a high performance compute system and Dr. Atsushi Kajita from [Fixstars Solutions Inc.](https://us.fixstars.com) for his leadership in software optimization using GPUs and SSDs. Together, we have transformed a rough codebase, that originally took days to run for a single field of view, into a powerful and robust software system that can process an ExSeq field of view within an hour. This software tool successfully processed terabytes of data successfully and automatically. This software pipeline has been a foundation for experimental productivity and biological exploration, and we hope it can be of value to labs around the world.

We thank all the people that have contributed to this codebase:

From Kioxia: Yosuke Bando, Shintaro Sano and Seiji Maeda.  

From FixStars: Atsushi Kajita, Karl Marrett, Ramdas Pillai, Robert Prior

From MIT: Dan Goodwin, Shahar Alon, Andrew Xue, Adam Marblestone, Anu Sinha, Oz Wassie

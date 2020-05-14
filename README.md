# ExSeqProcessing

This is the software library for processing Expansion Sequencing, ExSeq, experiments (Alon et al 2020). This pipeline takes multi-round in situ sequencing data (including morphological information for spatial context) and convert it into RNA reads in 3D. This software has succesfully processed over 300 fields of view of ExSeq data, corresponding to dozens of terabytes of data, and has helped illucidate biological phenomena at the nanoscale in neuroscience and between cell types in metastatic breast cancer. 

# Getting Started
Please refer to the [Wiki](https://github.com/dgoodwin208/ExSeqProcessing/wiki) for information and a tutorial, complete with a sample ExSeq dataset. 

We are grateful for all requests and questions - there will surely be bugs and we want to fix them. It is important to us that this software is a useful contribution to the community!

# Overview
In order to use this pipeline, your data must be formatted in way that can be ingested. You must create grayscale 3D images (either tiff or hdf5) and place them into a folder structure [explained here](https://github.com/dgoodwin208/ExSeqProcessing/wiki/Input-Data-Preparation). 

# Acknowledgements
This software pipeline has been a successful multi-year, multi-team collaboration. Specifically, the Boyden Lab would like to express gratitude to Dr. Yosuke Bando from [Kioxia](https://www.kioxia.com/en-us/top.html) (formerly Toshiba Memory) for his leadership in building a high performance compute system and Dr. Atsushi Kajita from [Fixstars Solutions Inc.](https://us.fixstars.com) for his leadership in software optimization using GPUs and SSDs. Together, we have transformed a rough codebase, that originally took days to run for a single field of view, into a powerful and robust software system that can process an ExSeq field of view within an hour. This software tool successfully processed terabytes of data successfully and automatically. This software pipeline has been a foundation for experimental productivity and biological exploration, and we hope it can be of value to labs around the world.

We thank all the people that have contributed to this codebase:

From Kioxia: Yosuke Bando, Shintaro Sano and Seiji Maeda.  

From FixStars: Atsushi Kajita, Karl Marrett, Ramdas Pillai, Robert Prior

From MIT: Dan Goodwin, Shahar Alon, Andrew Xue, Adam Marblestone, Anu Sinha, Oz Wassie

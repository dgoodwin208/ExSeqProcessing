# ImarisReader
ImarisReader is a set of MATLAB classes to read image and segmentation object data stored in [Imaris](http://www.bitplane.com/) .ims files.

# Introduction
Imaris .ims files are based on the HDF5 format. ImarisReader uses the HDF5 access functions built into MATLAB to read the data stored in ims files. ImarisReader encapsulates a set of MATLAB classes that facilitate access to the image data (DataSetReader) and segmented object data (CellsReader, FilamentsReader, SpotsReader and SurfacesReader).

# Install

Download the files, unzip, and then add the unzipped folder to the MATLAB path.

# Usage

**Create a reader object to read 'file.ims'.**
    
    fileObj = ImarisReader('file.ims');

**Read image volumes.**

    Imaris stores the image data in the Data
    vol = fileObj.DataSet.GetDataVolume(cIdx, tIdx);

* cIdx is the zero-based index for a channel
* tIdx is the zero-based index for a time point

**Read vertices and faces for a surface.**
    
    vertices = fileObj.Surfaces(1).GetVertices(sIdx);
    faces = fileObj.Surfaces(1).GetTriangles(sIdx);

* sIdx is the zero-based index for a surface

**Read spot positions.**
    
    pos = fileObj.Spots(1).GetPositions;

© 2016, Peter Beemiller (pbeemiller@gmail.com)

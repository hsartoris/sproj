/*! \file 
*
*/
#pragma once
#ifdef BUILD_HDF5

#include <string>
#include "hdf5.h"
#include <vector>
#include <iostream>

/*! \class hdf5Reader
 * \brief A class to handle reading hdf5 files.
 *
 * This class can read integer or floating point datasets.
 */
class hdf5Reader{
public:
/*! \brief Default constructor
 *
 */
    hdf5Reader();

/*! \brief Read integer dataset
 *
 * Read an integer dataset from an hdf5 file.
 * \param fname path to file
 * \param dname name of dataset in file
 * \param Mdata a reference to contain the output data
 *
 * For the file connectivity.h5, the viable datasetnames (and their types) are:
 * - mcN_layers (int)
 * - mcN_types (int)
 * - mcN_x (floating point)
 * - mcN_y (floating point)
 * - mcN_z (floating point)
 *
 * where N can be any integer between 0 and 6. For example, one could run the following code:<br>
 * std::string fname("connectivity.h5");<br>
 * std::string dname("mc1_layers");<br>
 * std::vector<int> layer_data;<br>
 * ReadIntDatasetFromFile(fname, dname, layer_data);
 */
    void ReadIntDatasetFromFile(std::string fname, std::string dname, std::vector<int>& Mdata);
    void ReadDoubleDatasetFromFile(std::string fname, std::string dname, std::vector<double>& Mdata);
private:
    std::string filename;
    std::string datasetname;
    herr_t status;
};

hdf5Reader::hdf5Reader() : filename("filename unset"), datasetname("datsetname unset") {};

void hdf5Reader::ReadIntDatasetFromFile(std::string fname, std::string dname, std::vector<int>& Mdata){
    filename = fname;
    datasetname = dname;
    /* identifiers */
    hid_t file_id, dataset_id;
    /* Open an existing file. */
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, datasetname.c_str(), H5P_DEFAULT);
    /*get info on data size*/
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    if(ndims != 1) { std::cerr << "You are not trying to read a 1D array!" ; }
    hsize_t dims[ndims];
    hsize_t maxdims[ndims];
    H5Sget_simple_extent_dims(dataspace_id,dims,maxdims);
   /* Allocate memory space*/
    Mdata.resize(dims[0]);
    /* read data */
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Mdata[0]);
    /* Close the dataset. */
    status = H5Dclose(dataset_id);
    /* Close the file. */
    status = H5Fclose(file_id);
}

void hdf5Reader::ReadDoubleDatasetFromFile(std::string fname, std::string dname, std::vector<double>& Mdata){
    filename = fname;
    datasetname = dname;
    /* identifiers */
    hid_t file_id, dataset_id;
    /* Open an existing file. */
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    dataset_id = H5Dopen(file_id, datasetname.c_str(), H5P_DEFAULT);
    /*get info on data size*/
    hid_t dataspace_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(dataspace_id);
    if(ndims != 1) { std::cerr << "You are not trying to read a 1D array!" ;}
    hsize_t dims[ndims];
    hsize_t maxdims[ndims];
    H5Sget_simple_extent_dims(dataspace_id,dims,maxdims);
    /* Allocate memory space*/
    Mdata.resize(dims[0]);
    /* read data */
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Mdata[0]);
    /* Close the dataset. */
    status = H5Dclose(dataset_id);
    /* Close the file. */
    status = H5Fclose(file_id);
}

#endif


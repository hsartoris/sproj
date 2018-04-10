#include "hdf5wrapper.h"
#include <iostream>
#include <cstdlib>


sparseSquareBinaryMatrix* readMatrixFromH5File( char* nameofH5File , char* nameOfMatrix )
{
    hdf5_wrapper h5(nameofH5File);
    unsigned nelems = h5.get_dataset_size(nameOfMatrix);
    std::cout << "Size is " << nelems << std::endl;
    hdf5_wrapper::bool_type * Bdata;
    Bdata = (hdf5_wrapper::bool_type *)malloc(nelems*nelems *sizeof(hdf5_wrapper::bool_type) );
    h5.read_bool_matrix_dataset(nameOfMatrix ,Bdata);


    sparseSquareBinaryMatrix* connections = new sparseSquareBinaryMatrix( nelems );
    unsigned positionNumber = 0;
    for ( unsigned i = 0 ; i != nelems ; ++i )
    {
        for ( unsigned j = 0 ; j != nelems ; ++j )
        {

            if ( i == j )continue; //removing 1's from the diagonal!!!

            if ( Bdata[positionNumber] )
            {
                connections->set( i,j );
            }
            ++positionNumber;
        }
    }

    return connections;
}

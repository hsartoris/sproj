/*    This file is part of the Neurotop Library. The Neurotop library
 *    allows to analyze higher topological structure, like simplices
 *    and Bett numbers of directed and not directed graphs.
 *    library for computational topology.
 *
 *    Author(s):       Pawel Dlotko
 *
 *    Copyright (C) 2015  INRIA Sophia-Saclay (France)
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


/*
* This code can operate on .h5 files, however since the library is not standard, in order to use .h5's please uncomment the line below. It will enable compilation of parts of code
* that are using .h5 library.
*/
//#define H5Header

using namespace std;

#include "SparseSquareBinaryMatrix.h"
#include "simplicialComplexSymmetry.h"
#include "computeHomology.h"

using namespace std;


int main( int argc, char** argv )
{
    if ( argc == 1 )
    {
        cout << "Wrong usage of a program. Please specify the input matrix i.e. call ./symmetric matrix (with .mat, .csv or .h5 ext). The second parameter is a 0/1 parameter. If 0, homology will not be computed. If 1, the homology will be computed.\n";
        cout << "The last (non obligatory) parameter is a 0/1 parameter. If 0, the list of simplices will be outpute. If 1, it will be outputed. Program will now terminate.";
        return 0;
    }

    bool computeHomolgy = (bool)atoi(argv[2]);
    cerr << "Will we compute homology : " << computeHomolgy << endl;

    bool shallWeOutputTheComplex = false;
    if ( argc > 3 )
    {
        int a = atoi( argv[3] );
        if ( a != 0 )
        {
            shallWeOutputTheComplex = true;
        }
    }

    sparseSquareBinaryMatrix* connections = new sparseSquareBinaryMatrix( argv[1] , true );
    connections->symmetrize();

    //creation o a simplcial complex. Look to files simplicialCOmplexSymmetry.h and simplxSymmetry.h
    simplicialComplex* cmplx = new simplicialComplex(*connections);


    /*
    * In below, by default we are outputing the number of simplices and the Euler characteristic to a file.
    */
	ofstream out;
	ostringstream name;
    name << argv[1] << "_symmetric_output.txt";
    std::string nameStr  = name.str();
    const char* filename1 = nameStr.c_str();
    out.open( filename1 );
    int eulerChar = 0;
    int a = 1;
	for ( size_t dim = 0 ; dim != cmplx->elemen().size() ; ++dim )
	{
 		out << cmplx->elemen()[dim].size() << endl;
		eulerChar += a*cmplx->elemen()[dim].size();
        a *= -1;
	}
	out << "Euler characteristic : " << eulerChar << endl;
	cout << "Euler characteristic : " << eulerChar << endl;


    /*
    * In this case, once the user set up the the third parameter of the program call to 1, then all the simplices will be written to a file. They will be stored as sequence of numbers,
    * where each number correspond to number of a column of the input matrix.
    */
    if ( shallWeOutputTheComplex )
    {
        for ( size_t dim = 0 ; dim != cmplx->elemen().size() ; ++dim )
        {
            ostringstream name;
            name << argv[1] << "_simplices_dimension_" << dim << ".txt";
            std::string nameStr  = name.str();
            const char* filename1 = nameStr.c_str();
            ofstream out;
            out.open( filename1 );
            for ( size_t i = 0 ; i != cmplx->elemen()[dim].size() ; ++i )
            {
                out << *cmplx->elemen()[dim][i] << endl;
            }
            out.close();
        }
    }

    /*
    * In this case, once the user set up the the second parameter of the program call to 1, then we will compute homology of the complex. Note that the worst case complexity of this procedure
    * is quadratic (with respect to the total number of simplices in the complex) and therefore this procedure may take in some cases (really) a lot of time to execute.
    */
    if ( computeHomolgy )
    {
        std::vector< size_t > bettiNumbers = computeHomology( cmplx );
        out << "Standard Betti numbers: ";
        for ( size_t i = 0 ; i != cmplx->elemen().size() ; ++i )
        {
            out << bettiNumbers[i] << " ";
        }
    }
    return 0;
}

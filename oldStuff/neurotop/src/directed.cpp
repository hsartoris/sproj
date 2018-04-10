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

//uncomment the line below if you would like to be able to read .h5 files:
//#define H5Header


#include "simplex.h"
#include "SparseSquareBinaryMatrix.h"
#include "simplicialComplex.h"
#include "computeHomology.h"
using namespace std;


int main( int argc, char** argv )
{
    if ( argc == 1 )
    {
        cout << "Wrong usage of a program. Please specify the input matrix i.e. call ./directed matrix (having .mat, .csv or .h5 ext). The second parameter is a 0/1 parameter. If 0, homology will not be computed. If 1, the homology will be computed. Program will now terminate.";
        return 0;
    }

    bool computeHomology_ = (bool)atoi(argv[2]);
    cerr << "Will we compute homology : " << computeHomology_ << endl;

    bool shallWeOutputTheComplex = false;
    if ( argc > 3 )
    {
        int a = atoi( argv[3] );
        if ( a != 0 )
        {
            shallWeOutputTheComplex = true;
        }
    }
    sparseSquareBinaryMatrix connections( argv[1] , false );
    simplicialComplex* cmplx = new simplicialComplex(connections);



	ostringstream name;
    name << argv[1] << "_directed_output.txt";
    std::string nameStr  = name.str();
    const char* filename1 = nameStr.c_str();
    ofstream output;
    output.open(filename1);

	int eulerChar = 0;
    int a = 1;
	for ( size_t dim = 0 ; dim != cmplx->elemen().size() ; ++dim )
	{
		output  << cmplx->elemen()[dim].size() << endl;
		eulerChar += a*cmplx->elemen()[dim].size();
        a *= -1;
	}
    output << "Euler characteristic : " << eulerChar << endl;
    cout << "Euler characteristic : " << eulerChar << endl;


    if ( shallWeOutputTheComplex )
    {
        cerr << "Outputting complex to a file \n";
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

    //homology computation part:
    if ( computeHomology_    )
    {
        std::vector< size_t > bettiNumbers = computeHomology( cmplx );
        output << "Standard Betti numbers: ";
        for ( size_t i = 0 ; i != cmplx->elemen().size() ; ++i )
        {
            output << bettiNumbers[i] << " ";
        }
    }
    return 0;
}

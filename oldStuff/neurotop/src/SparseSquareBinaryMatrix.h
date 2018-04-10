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


#pragma once

#include <string>
#include <vector>
#include <set>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <sstream>
#ifdef BUILD_HDF5
    #include "hdf5.h"
#endif

using namespace std;

/*
* This is a sparseSquareBinaryMatrix, simple implementation of a sparse matrix data structure. The input to the construction of symmetric or directed simplicial complex is a matrix
* stored in one of given file formats (binary, csv, h5, with a possible extension in the future). The matrix is stored as a vector of sets. The vector contains as many elements as
* the number of columns in the matrix. Each set represent a column. Since in this case matrices are binary, in the set representing a column we store id's of rows that are nonzero
* in the matrix.
*/
class sparseSquareBinaryMatrix
{
public:

    /*
    * This is a the main constructor sparseSquareBinaryMatrix class. It takes as an input the name of a file. We assume that this is the file with one of the following extensions:
    * .bin -- for binary files
    * .csv, for comma separated value files
    * .h5 for h5 files.
    * The second parameter of a constructor is a boolean value is_symmetric. If we want to construct symmetric matrix (even in case when the input matrix is not symmetric), this
    * parameter have to be set to true. If it is set to false, then the constructed matrix is 1-1 as the matrix in the file.
    * The last two parameters of the constructor are needed only for .h5 files. The sizeOfAMatrix parameter tells the size of the square matrix in .h5 file.
    * The nameOfMatrix parameter tells what is the name of the matrix we want to get in the .h5 file.
    * If the matrix is to be used to build a simplicial complex, we assume that there is nothing nonzero in the diagonal.
    */
    sparseSquareBinaryMatrix(const char* filename , bool is_symmetric_ ,  int sizeOfAMatrix = 31346 , char* nameOfMatrix = (char*)"matrix_0")
    {
        this->is_symmetric = is_symmetric_;
        this->readConnectionMatrix(filename,sizeOfAMatrix ,nameOfMatrix);
    };

    /*
    * Constructor of a empty matrix.
    */
    sparseSquareBinaryMatrix(int size):size_(size)
    {
        srand( time(0) );
        std::vector< std::set<size_t> > vect(size);
        this->nonzeroPositions = vect;
    };

    /*
    * Operator (x,y) returns the value at the position (x,y) of the matrix.
    */
    bool operator()(unsigned x, unsigned y)
    {
        if ( this->nonzeroPositions[x].find(y) == this->nonzeroPositions[x].end() ) return 0;
        return 1;
    }

    /*
    * Operator (i) returns the set representing the i-th column of the matrix.
    */
    std::set<size_t> operator()(int i)
    {
        return this->nonzeroPositions[i];
    }

    /*
    * Operator (i) that write the matrix to a stream. The matrix is written in a sparse form, i.e. only nonzero elements in columns are in the stream.
    * Different columns can be recognized by the string "Nonzero elements in the column : ".
    */
    friend ostream& operator << ( ostream& out , const sparseSquareBinaryMatrix& m )
    {
        for ( size_t collNo = 0 ; collNo != m.nonzeroPositions.size() ; ++collNo )
        {
            if ( m.nonzeroPositions[collNo].size() == 0 )continue;
            out << "Nonzero elements in the column : " << collNo << ": ";
            for ( std::set<size_t>::iterator it = m.nonzeroPositions[collNo].begin() ; it != m.nonzeroPositions[collNo].end() ; ++it )
            {
                out << *it << " ";
            }
            out << endl;
        }
        return out;
    }

    /*
    * The function set(x,y) sets the value of position x,y of the matrix to true.
    */
    inline void set( unsigned x , unsigned y )
    {
        if ( (x >= this->size_) || (y >= this->size_) )throw "Wrong values of (x,y) in the set(x,y) procedure in the sparseSquareBinaryMatrix class.";
        this->nonzeroPositions[y].insert(x);
    }

    /*
    * The procedure takeRandomSubmatrix returns random submatrox of the given matrix. The size of the submatrix is determined by the numberOfColumnsInSubsample.
    */
    sparseSquareBinaryMatrix* takeRandomSubmatrix( size_t numberOfColumnsInSubsample );

    /*
    * The procedure numberOfConnections returns total number of nonzero elements in matrix. If the matrix is interpreted as graph, this number is the number of connections for
    * not symmetric matrices. For symmetric matrices it is twice the number of connections (since each connection in a graph corresponds to two 'true' values in the matrix.
    */
    size_t numberOfConnections();

    /*
    * The procedure symmetrize change the given matrix into symmetric one.
    */
    void symmetrize();

    /*
    * The procedure size return the size of the matrix (i.e. number of rows or number of columns).
    */
    size_t size(){return this->size_;}


    /*
    * The procedure store_as_csv stores given matrix as csv file named as the char* filename.
    */
    void store_as_csv( const char* filename );

private:
    bool is_symmetric;
    unsigned size_;
    std::vector< std::set<size_t> > nonzeroPositions;
    void readConnectionMatrix( const char* filename  ,  int sizeOfAMatrix = 31346 , char* nameOfMatrix = (char*)"matrix_0");
};


void sparseSquareBinaryMatrix::readConnectionMatrix( const char* filename  ,  int sizeOfAMatrix, char* nameOfMatrix )
{
    bool dbg = false;

    bool didWeReadMatrix = false;
    //if the file is a binary file.
    if ( ( filename[ strlen( filename )- 1 ] == 'n' ) && ( filename[ strlen( filename )- 2 ] == 'i' ) && ( filename[ strlen( filename )- 3 ] == 'b' ) )
    {
        didWeReadMatrix = true;
        cerr << "This is a binary file : " << filename << endl;
        ifstream mat( filename , ios::binary | ios::ate );
        unsigned size = mat.tellg();
        mat.close();
        unsigned sideSize = sqrt( size );
        //numberOfAllVetices = sideSize;


        this->size_ = sideSize;
        mat.open(filename,ios::binary);

        int numberNonzero = 0;
        for ( unsigned i = 0 ; i != sideSize ; ++i )
        {
            int numberOfNonZeroInCollumn = 0;
            for ( unsigned j = 0 ; j != sideSize ; ++j )
            {
                if ( i == j )continue; //removing 1's from the diagonal!!!

                char val;
                mat.get(val);
                if ( val )
                {
                    this->set(i,j);
                    ++numberNonzero;
                    ++numberOfNonZeroInCollumn;
                }
            }
        }
        mat.close();
    }


    if ( ( filename[ strlen( filename )- 1 ] == '5' ) && ( filename[ strlen( filename )- 2 ] == 'h' )  )
    {
        didWeReadMatrix = true;
        cerr << "This is a h5 file " << filename << endl;
        cerr << "Reading matrix : " << nameOfMatrix << endl;
        #ifdef H5Header
        return readMatrixFromH5File( filename , nameOfMatrix );
        #endif
    }

    //if the file is a csv file.
    if ( ( filename[ strlen( filename )- 1 ] == 'v' ) && ( filename[ strlen( filename )- 2 ] == 's' ) && ( filename[ strlen( filename )- 3 ] == 'c' ) )
    {
        didWeReadMatrix = true;
        cerr << "This is a csv file " << filename << endl;
         //in this case, we have csv like file
        stringstream buffer;
        std::ifstream mat( filename );
        buffer << mat.rdbuf();
        mat.close();

        //now go through whole buffer and change ',' to ' ':
        std::string  bufferStr = buffer.str();

        size_t number_of_elements = 1;
        for ( size_t i = 0 ; i != bufferStr.size() ; ++i )
        {
            if ( bufferStr[i] == ',' )
            {
                bufferStr[i] = ' ';
                ++number_of_elements;
            }
        }

        std::stringstream ss;
        ss << bufferStr;


        std::vector<bool> elems;
        elems.reserve( number_of_elements );
        while ( ss.good() )
        {
            bool val;
            ss >> val;
            elems.push_back( val );
            if (dbg)cerr << "Reading : " << val << endl;
        }


        size_t numberOfAllVetices = (size_t)sqrt( elems.size() );
        size_t sideSize = numberOfAllVetices;
        this->size_ = sideSize;

        this->nonzeroPositions = std::vector< std::set<size_t> >(sideSize);
        mat.open(filename);

        int numberNonzero = 0;
        int aa = 0;
        for ( unsigned i = 0 ; i != sideSize ; ++i )
        {
            for ( unsigned j = 0 ; j != sideSize ; ++j )
            {
                if ( (i != j ) && (elems[aa]) )//ignoring zeros on diagonal
                {
                    if ( this->is_symmetric )
                    {
                        if ( ((*this)(j,i) == false) || ((*this)(i,j) == false) )
                        {
                                this->set(j,i);
                                this->set(i,j);
                        }
                    }
                    else
                    {
                        this->set(j,i);
                    }
                    ++numberNonzero;
                }
                ++aa;
            }
        }

        cout << "Number of nonzero elements in the matrix : " << numberNonzero << endl;
    }

    if ( !didWeReadMatrix )
    {
        cerr << "The file extension is not supported. Program will now terminate \n";
        throw "The file extension is not supported. Program will now terminate \n";
    }
}


size_t sparseSquareBinaryMatrix::numberOfConnections()
{
    size_t result = 0;
    for ( size_t i = 0 ; i != this->nonzeroPositions.size() ; ++i )
    {
        result += this->nonzeroPositions[i].size();
    }
    return result;
}

void sparseSquareBinaryMatrix::symmetrize()
{
    for ( size_t collNo = 0 ; collNo != this->nonzeroPositions.size() ; ++collNo )
    {
        for ( std::set<size_t>::iterator it = this->nonzeroPositions[collNo].begin() ; it != this->nonzeroPositions[collNo].end() ; ++it )
        {
            this->set( *it,collNo );
            this->set( collNo,*it );
        }
    }
}


sparseSquareBinaryMatrix* sparseSquareBinaryMatrix::takeRandomSubmatrix( size_t numberOfColumnsInSubsample )
{
    bool dbg = true;

    if ( numberOfColumnsInSubsample > this->size_ )throw("Matrix is smaller than numberOfColumnsInSubsample. Program terminate");

    std::set< size_t > choosenCollumns;

    while ( choosenCollumns.size() != numberOfColumnsInSubsample )
    {
        choosenCollumns.insert( rand() % this->size() );
    }

    if ( dbg )
    {
        cerr << "Here are the numbers of columns/rows we are going to use : " << endl;
        for ( std::set< size_t >::iterator it = choosenCollumns.begin() ; it != choosenCollumns.end() ; ++it )
        {
            cerr << *it << " , ";
        }
        cerr << endl;
    }

    sparseSquareBinaryMatrix* result = new sparseSquareBinaryMatrix( this->size() );

    for ( std::set<size_t>::iterator it = choosenCollumns.begin() ; it != choosenCollumns.end() ; ++it )
    {
        //read the column it from the matrix this
        for ( std::set<size_t>::iterator elementsInCol = this->nonzeroPositions[*it].begin() ; elementsInCol != this->nonzeroPositions[*it].end() ; ++elementsInCol )
        {
            if ( choosenCollumns.find(*elementsInCol) != choosenCollumns.end() )
            {
                result->set( *it , *elementsInCol );
            }
        }
    }
    return result;
}

void sparseSquareBinaryMatrix::store_as_csv( const char* filename )
{
    ofstream out;
    out.open( filename );
    for ( size_t row = 0 ; row != this->size() ; ++row )
    {
        for ( size_t col = 0 ; col != this->size() ; ++col )
        {
            if ( this->nonzeroPositions[col].find( row ) != this->nonzeroPositions[col].end() )
            {
                out << "1";
            }
            else
            {
                out << "0";
            }
            if ( col != this->size()-1 )out << ",";
        }
        out << std::endl;
    }
    out.close();
}//store_as_csv

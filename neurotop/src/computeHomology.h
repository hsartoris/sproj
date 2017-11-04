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

// wrapper algorithm that computes the persistence pairs of a given boundary matrix using a specified algorithm
#include "../include/phat/compute_persistence_pairs.h"

// main data structure (choice affects performance)
#include "../include/phat/representations/vector_vector.h"

// algorithm (choice affects performance)
#include "../include/phat/algorithms/standard_reduction.h"
#include "../include/phat/algorithms/chunk_reduction.h"
#include "../include/phat/algorithms/row_reduction.h"
#include "../include/phat/algorithms/twist_reduction.h"


std::vector< size_t > computeHomology( simplicialComplex* cmplx )
{
        phat::boundary_matrix< phat::vector_vector > boundary_matrix;

        int numberOfCells = 0;
        for ( size_t dim = 0 ; dim != cmplx->elements.size() ; ++dim )
            for ( size_t i = 0 ; i != cmplx->elements[dim].size() ; ++i )
                if ( !cmplx->elements[dim][i]->deleted() ) ++numberOfCells;


        // set the number of columns (has to be 7 since we have 7 simplices)
        boundary_matrix.set_num_cols( numberOfCells );

        // set the dimension of the cell that a column represents:
        int nrOfGenerator = 0;

        simplex** numberOfGeneratorToDim = new simplex*[numberOfCells+1];


        int* bettiNumbers = new int[cmplx->elements.size()];
        for ( size_t i = 0 ; i != cmplx->elements.size() ; ++i )
            bettiNumbers[i] = 0;
        for ( size_t dim = 0 ; dim != cmplx->elements.size() ; ++dim )
        {
            for ( size_t i = 0 ; i != cmplx->elements[dim].size() ; ++i )
            {
                if ( cmplx->elements[dim][i]->deleted() )continue;
                bettiNumbers[dim]++;
                boundary_matrix.set_dim( nrOfGenerator, dim );
                cmplx->elements[dim][i]->number = nrOfGenerator;
                numberOfGeneratorToDim[nrOfGenerator] = cmplx->elements[dim][i];
                ++nrOfGenerator;
            }
        }

        // set the respective columns -- the columns entries have to be sorted
        std::vector< phat::index > temp_col;
        for ( size_t dim = 0 ; dim != cmplx->elements.size() ; ++dim )
        {
            for ( size_t i = 0 ; i != cmplx->elements[dim].size() ; ++i )
            {
                if ( cmplx->elements[dim][i]->delet )continue;

                simplex* aa = cmplx->elements[dim][i];
                size_t numberElInBd = 0;
                for ( simplex::BdIterator bd = aa->bdBegin() ; bd != aa->bdEnd() ; ++bd )
                {
                    if ( (*bd)->delet )continue;
                    temp_col.push_back( (*bd)->number );
                    numberElInBd++;
                }

                if ( (dim > 0)&&(numberElInBd != dim+1) )
                {
                    std::cerr << *cmplx->elements[dim][i] << std::endl;
                    std::cerr << "something is wrong with this simplex \n";
                    std::cerr << "dim : " << dim << std::endl;
                    std::cerr << "numberElInBd : " << numberElInBd << std::endl;
                    getchar();
                }

                std::sort( temp_col.begin() , temp_col.end() );
                boundary_matrix.set_col( cmplx->elements[dim][i]->number , temp_col );
                temp_col.clear();
            }
        }


        // define the object to hold the resulting persistence pairs
        phat::persistence_pairs pairs;



        // choose an algorithm (choice affects performance) and compute the persistence pair
        // (modifies boundary_matrix)
        //phat::compute_persistence_pairs< phat::chunk_reduction >( pairs, boundary_matrix );


        phat::compute_persistence_pairs_dualized< phat::chunk_reduction >( pairs, boundary_matrix );


        for( phat::index idx = 0; idx < pairs.get_num_pairs(); idx++ )
        {
            //std::cout << "Birth: " << pairs.get_pair( idx ).first << ", Death: " << pairs.get_pair( idx ).second << std::endl;
            simplex* first = numberOfGeneratorToDim[pairs.get_pair( idx ).first];
            simplex* second = numberOfGeneratorToDim[pairs.get_pair( idx ).second];
            bettiNumbers[ first->dim() ]--;
            bettiNumbers[ second->dim() ]--;
        }

        std::vector< size_t > result;
        for ( size_t i = 0 ; i != cmplx->elements.size() ; ++i )
        {
            result.push_back( bettiNumbers[i] );
        }
        return result;
}

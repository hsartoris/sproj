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


#include <vector>
#include <iostream>
#include <set>
#include <fstream>

class simplicialComplex;

bool dispalyInfo = false;


class simplex;
class simplicialComplex;
void setUpVerticesInThisSimplex( simplex* tri , std::vector< simplex* > facesOfThisSimplex );
std::vector< int > howManyMultipleTriangelsThereAre( simplicialComplex * cmplx );
std::vector< int > howManyMultipleTriangelsThereAre( simplicialComplex * cmplx );


/*
* This is a directed simplex. A directed simplex consist of a sequence of vertices (order matters!).
*/
class simplex
{
public:
    /*
    * Default constructor of a simplex.
    */
    simplex();

    /*
    * This procedure compute intersection of two simplices (two sequences of vertices).
    */
    void intersection( simplex* t1,simplex* t2);

    /*
    * Iterator for boundary of a simplex.
    */
    typedef  std::vector< simplex* >::const_iterator BdIterator;

    /*
    * Iterator for a coboundary of a simplex.
    */
    typedef  std::vector< simplex*>::const_iterator CbdIterator;


    const BdIterator bdBegin()
    {
          return this->bound.begin();
    };
    const BdIterator bdEnd()
    {
          return this->bound.end();
    };
    const CbdIterator cbdBegin()
    {
          return this->coBound.begin();
    };
    const CbdIterator cbdEnd()
    {
          return this->coBound.end();
    };

    /*
    * This procedure marks a simpelx as deleted. It can be used by elementaryReduction() or coreduction() or other methods. Note that deleting a simplex do not mean removing it from a structure, just marking it as deleted.
    */
    inline void del(){this->delet = true;};

    /*
    * A method to delete/undelete a simplex. Note that deleting a simplex do not mean removing it from a structure, just marking it as deleted.
    */
    inline bool& deleted(){return this->delet;};

    /*
    * Operator that write simplex to a stream. Vertices are written in the order they appear in the simplex.
    */
    friend std::ostream& operator<<(std::ostream& out, simplex& sim)
    {
         if ( sim.dim() > 0 )
         {
               std::vector< simplex*,
              std::allocator<simplex*> >::iterator it;
              for ( it = sim.verticesInThisSimplex.begin() ;
                    it != sim.verticesInThisSimplex.end() ; ++it )
              {
                    out << (*it)->numberInCmplx() << " ";
              }
         }
         else
         {
             out << sim.numberInComplex;
         }
         return out;
    };

    /*
    * Procedure that returns number of the simplex in the complex (this number is set up by suitable simplicialComplex constructor.
    */
    inline unsigned int numberInCmplx() const {return this->numberInComplex;}

    /*
    * Comparation of simplices (based on their numbers in the complex).
    */
    inline friend bool operator < ( const simplex& t1, const simplex& t2 )
    {
           return( t1.numberInComplex < t2.numberInComplex );
    }

    /*
    * Comparation of simplices (based on their numbers in the complex).
    */
    inline friend bool operator == ( const simplex& t1, const simplex& t2 )
    {
           return( t1.numberInComplex == t2.numberInComplex );
    }

    /*
    * Procedure returning dimension of a simplex.
    */
    inline int dim(){return this->verticesInThisSimplex.size() - 1;}

    /*
    * Procedure that returns vertices in the current simplex (in the right order).
    */
    std::vector<simplex*> vertices(){return this->verticesInThisSimplex;}

    friend class simplicialComplex;
    friend std::vector< size_t > computeHomology( simplicialComplex* cmplx );
    friend std::vector<simplex * > intersect( std::vector<simplex * >& f , std::vector<simplex * >& s ,unsigned a, unsigned b);
    friend void setUpVerticesInThisSimplex( simplex* tri , std::vector< simplex* > facesOfThisSimplex );
    friend std::vector< int > howManyMultipleTriangelsThereAre( simplicialComplex * cmplx );

protected:
    unsigned number;
    unsigned long int numberInComplex;
    std::vector<simplex*> bound;
    std::vector<simplex*> coBound;
    std::vector<simplex*> elementsNotFurtherThanEpsilon;
    std::vector<simplex*> verticesInThisSimplex;
    bool delet;
};//simplex

/*
* Comparation of simplices (based on their numbers in the complex).
*/
inline bool same(simplex*a,simplex*b){return (a->numberInCmplx() == b->numberInCmplx());}

simplex::simplex()
{
     this->numberInComplex = -1;
     this->delet = false;
     this->number = 0;
}

const int higherDebug = 0;
void simplex::intersection(simplex* t1,simplex* t2)
{
    //since we are having some problems with the code, to make sure all is OK, I am using this
    //nlogn procedure instead of linear time procedure presented in below.
    std::set< simplex* > s;
    for ( size_t v1 = 0 ; v1 != t1->elementsNotFurtherThanEpsilon.size() ; ++v1 )
    {
        s.insert( t1->elementsNotFurtherThanEpsilon[v1] );
    }
    for ( size_t v2 = 0 ; v2 != t2->elementsNotFurtherThanEpsilon.size() ; ++v2 )
    {
        if ( s.find( t2->elementsNotFurtherThanEpsilon[v2] ) != s.end() )
        {
            this->elementsNotFurtherThanEpsilon.push_back(t2->elementsNotFurtherThanEpsilon[v2]);
        }
    }
}//intersection

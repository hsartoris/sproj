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


#include <sstream>
#include "simplexSymmetry.h"

/*
* A comparision function for two simplices.
*/
bool comparision( simplex* s1, simplex* s2 )
{
    return (s1->numberInCmplx() < s2->numberInCmplx());
}

class simplicialComplex;
std::vector< size_t > computeHomology( simplicialComplex* cmplx );

/*
* This is a simplicial complex class. A simplicial complex is a collection of simplices of different dimensions having the property that every subset of (vertices) of a simplex
* that belongs to an abstract simplicial complex is a simplex in that complex.
*/
class simplicialComplex
{
public:
       simplicialComplex( sparseSquareBinaryMatrix& mat , int chopDimnesion = -1 );
       simplicialComplex();


       /*
       * This is a support function called by the constructor. Its purpose is to populate the \ref elements member with the list of simplices of dimension 0 and 1.
       * Simplices of dimension 0 are the nodes of the graph. Simplices of dimension 1 are the edges connecting the nodes of the graph.
       * For each column of the incidence matrix, a new simplex of dimension 0 is created and added to the std::vector of 0-dimensional simplices.
       * For each couple of nonzero entries in each column of the incidence matrix, a simplex of dimension 1 (an edge) is created.
       * Each simplex also stores, in a std::vector called \ref elementsNotFurtherThanEpsilon, the IDs of lower dimensional simplices that are in the intersection of all the coboundaries of each simplex that makes up its boundary. This is used to create higher dimensional simplexes later on.
       * At the end of execution, the \ref elements member will be populated with the 0 and 1 dimensional simplices. The \ref numberOfSimplexes member will be correspondingly updated.
       * \param mat an incidence matrix, i.e. a square binary matrix, where each row/column corresponds to a node of the graph, and the entry (i,j) is 1 if nodes i and j are connected, 0 otherwise.
       */
       void frame( sparseSquareBinaryMatrix& mat );


       /*
       * This is a support function called by the constructor. Its purpose is to populate the \ref elements member with the list of simplices of dimension higher than 1.
       * In order to generate the list of simplices of dimension n, we loop over all the simplices of dimension n-1. For each of these simplices, we loop over the simplices in the boundary (of dimension n-2). For each simplex of the boundary, we loop over the simplices in the coboundary (of dimension n-1) and check if they contain the simplices in the list \ref elementsNotFurtherThanEpsilon. If they do, than they are a condidate to create a simplex of dimension n, and we store them in the \ref elements member.
       * If for a given dimension m no new simplices are created, we stop the process.
       * \return the number of created simplices for a given dimension.
       */
       int createHigherdimensionalSimplexes();

       /*
       * This is a debug purpose procedure to make sure that the complex is correct.
       */
       bool test( sparseSquareBinaryMatrix& mat );

       /*
       * This is a debug purpose procedure to make sure that the complex is correct.
       */
       bool test2inSearchForMaximalCliques( sparseSquareBinaryMatrix& mat );

       /*
       * Operator that writes simplicial complex to a stream. The output is formatted in the following way:
       * The simplices are put to the stream ordered by their dimension. There is no particular, relaiable order for simplices having the same dimension.
       */
       friend std::ostream& operator<<(std::ostream& out, const simplicialComplex& cmplx)
       {
           for ( unsigned int i = 0 ; i < cmplx.elements.size() ; ++i )
           {
               out << "\n dimension : " << i <<"\n";
               std::vector< simplex* > li = cmplx.elements[i];
                std::vector< simplex*,
               std::allocator<simplex*> >::iterator it;
               for ( it = li.begin() ; it != li.end() ; ++it )
               {
                   out << "\n" ;
                   out << (*(*it));
                   out <<"\n";
               }
           }
           return out;
      }

       /*
       * This procedure performs elementrary reductions aka free face collapses on the complex. The idea of free face collapses is to remove some cells of the complex keeping its
       * so called homotopy type. In particular, homology of the complex are not affected.
       */
      int elementaryReductions()
      {
          std::cerr << "Elementary reductions \n";
          int ilosc = 0;
          std::list<simplex*> freeFaces;
          for ( unsigned dim = 0 ; dim != this->elements.size()-1 ; ++dim )
          {
              //std::cerr <<"dim : " << dim << " ilosc : " << this->elements[dim].size() << std::endl;
              for (  std::vector< simplex* >::iterator it = this->elements[dim].begin() ; it != this->elements[dim].end() ; ++it )
              {
                  if ( (*it)->deleted() )continue;
                  int iloscELementowWKobrzegu = 0;

                  for (  std::vector<simplex*>::iterator cbd = (*it)->coBound.begin() ; cbd != (*it)->coBound.end() ; ++cbd )
                  {
                      if ( !(*cbd)->delet )++iloscELementowWKobrzegu;
                  }
                  if ( iloscELementowWKobrzegu == 1 )
                  {
                      freeFaces.push_back( *it );
                  }
              }
          }
          while ( !freeFaces.empty() )
          {
              simplex* top = *freeFaces.begin();
              freeFaces.pop_front();
              if ( top->delet )continue;

              int iloscELementowWBrzegu = 0;
              simplex* second = 0;
              for (  std::vector<simplex*>::iterator cbd = top->coBound.begin() ; cbd != top->coBound.end() ; ++cbd )
              {
                  if ( !(*cbd)->delet )
                  {
                      ++iloscELementowWBrzegu;
                      second = *cbd;
                  }
              }
              if ( (iloscELementowWBrzegu == 1) && (!top->delet) && (!second->delet) )
              {
                  ++ilosc;
                  top->delet = true;
                  second->delet = true;
                  for (  simplex::BdIterator bd = top->bdBegin() ; bd != top->bdEnd() ; ++bd )
                  {
                      if ( !(*bd)->delet ) freeFaces.push_back( *bd );
                  }
                  for (  simplex::BdIterator bd = second->bdBegin() ; bd != second->bdEnd() ; ++bd )
                  {
                      if ( !(*bd)->delet ) freeFaces.push_back( *bd );
                  }
              }
          }
          return ilosc;
      }


       /*
       * This procedure is for a debug purposes.
       */
      simplex* checkIfTheSimplesHavingFollowingVerticesExist( std::vector<unsigned> ver )
      {
          std::sort( ver.begin() , ver.end() );
          for ( size_t i = 0 ; i != this->elements[ver.size()-1].size() ; ++i )
          {
              bool theSame = true;
              for ( size_t nr = 0 ; nr != ver.size() ; ++nr )
              {
                  if ( ver[nr] != this->elements[ver.size()-1][i]->verticesInThisSimplex[nr]->numberInComplex )
                  {
                      theSame = false;
                      break;
                  }
              }
              if ( theSame )
              {
                  std::cout << "I have found a simplex \n";

                  std::cout << "ver.size() : " << ver.size() << std::endl;
                  std::cout << "i : " << i << std::endl;
                  std::cout << "this->elements[ver.size()-1].size() : " << this->elements[ver.size()-1].size() << std::endl;
                  std::cout << *this->elements[ver.size()-1][i] << std::endl;
                  return this->elements[ver.size()-1][i];
              }
          }
          return 0;
      }

       /*
       * This procedure performs so called coreductions, see
       * http://link.springer.com/article/10.1007%2Fs00454-008-9073-y
       * It remove some cells of the complex keeping its reduced homology unchanged. In particular, all homology of dimension greater than zero of the complex will not be affected by this procedure.
       */
      int coreductions()
      {
          for (  std::vector< simplex* >::iterator it = this->elements[0].begin() ; it != this->elements[0].end() ; ++it )
          {
              if ( !(*it)->delet )
              {
                  extern int betaZero;
                  ++betaZero;
                  (*it)->delet = true;
                  break;
              }
          }


          int ilosc = 0;
          std::list<simplex*> freeCoFaces;
          for ( unsigned dim = 0 ; dim != this->elements.size() ; ++dim )
          {
              for (  std::vector< simplex* >::iterator it = this->elements[dim].begin() ; it != this->elements[dim].end() ; ++it )
              {
                  if ( (*it)->delet )continue;
                  int iloscELementowWKobrzegu = 0;

                  for (  simplex::BdIterator bd = (*it)->bdBegin() ; bd != (*it)->bdEnd() ; ++bd )
                  {
                      if ( !(*bd)->delet )++iloscELementowWKobrzegu;
                  }
                  if ( iloscELementowWKobrzegu == 1 )
                  {
                      freeCoFaces.push_back( *it );
                  }
              }
          }


        //std::cerr << "freeFaces.size() " << freeFaces.size() << std::endl;

          while ( !freeCoFaces.empty() )
          {
              simplex* top = *freeCoFaces.begin();
              freeCoFaces.pop_front();
              if ( top->delet )continue;

              int iloscELementowWBrzegu = 0;
              simplex* second = 0;
              for (  std::vector<simplex*>::iterator bd = top->bound.begin() ; bd != top->bound.end() ; ++bd )
              {
                  if ( !(*bd)->delet )
                  {
                      ++iloscELementowWBrzegu;
                      second = *bd;
                  }
              }
              if ( iloscELementowWBrzegu == 1 )
              {
                  ++ilosc;

                  if ( (top->delet) || (second->delet) ){std::cerr << "Cos jest zjebane z koredukcjami \n";getchar();}

                  top->delet = true;
                  second->delet = true;
                  for (  simplex::CbdIterator cbd = top->cbdBegin() ; cbd != top->cbdEnd() ; ++cbd )
                  {
                      freeCoFaces.push_back( *cbd );
                  }
                  for (  simplex::CbdIterator cbd = second->cbdBegin() ; cbd != second->cbdEnd() ; ++cbd )
                  {
                      freeCoFaces.push_back( *cbd );
                  }
              }
          }
          //std::cerr << "Number of coreduced pairs : " << ilosc << std::endl;
          return ilosc;
      }


      /*
       *Destructor.
      */
      ~simplicialComplex()
      {
          if (deletedd)
          {
              return;
          }
          deletedd = true;
          for (  size_t i = 0 ; i != this->elements.size() ; ++i )
          {
            for ( size_t j = 0 ; j != this->elements[i].size() ; ++j )
                delete this->elements[i][j];
          }
      }

      /*
       * A procedure that writes maximal simplices to a given file. A simplex is said to be maximal if it is not in a boundary of any other simplex. Set of maximal simplices is typically
       * much smaller than the set of all simplices, yet, the whole complex can be re-created from the set of its maximal simplices.
      */
      void writeMaximalSimplicesToFile( char* filename );

      /*
       * A procedure that writes all simplices to a given file.
      */
      void writeToFile(char* filename);

       /*
       * Accessor to the list of simplices graded by dimension.
      */
      std::vector< std::vector< simplex* > >& elemen(){return this->elements;}

      friend std::vector< size_t > computeHomology( simplicialComplex* cmplx );
protected:

       //! short description
       bool deletedd;
       //! short description
       std::vector< std::vector< simplex* > > elements;
       //! short description
       unsigned int numberOfSimplexes;
       //! short description
       bool disposed;
};//simplicialComplex



void simplicialComplex::writeMaximalSimplicesToFile( char* filename )
{
    for ( size_t dim = 0 ; dim != this->elements.size() ; ++dim )
    {
        std::ofstream out;
        std::ostringstream name;
        name << filename << "_symmetric_maxSimpl_dim_" << dim << ".txt";
        std::string nameStr  = name.str();
        const char* filename1 = nameStr.c_str();
        out.open( filename1 );
        for ( size_t nr = 0 ; nr != this->elements[dim].size() ; ++nr )
        {
            if ( this->elements[dim][nr]->coBound.size() == 0 )
            {
                out << *this->elements[dim][nr] << std::endl;
            }
        }
        out.close();
    }
}//writeMaximalSimplicesToFile


void simplicialComplex::writeToFile(char* filename)
{
    for ( size_t dim = 0 ; dim != this->elements.size() ; ++dim )
    {
        std::ostringstream name;
        name << filename << "_directed_simplices_ " << dim << ".txt";
        std::string nameStr  = name.str();
        const char* filename1 = nameStr.c_str();
        std::ofstream output;
        output.open(filename1);

        for ( size_t nr = 0 ; nr != this->elements[dim].size() ; ++nr )
        {
            output << *this->elements[dim][nr] << std::endl;
        }

        output.close();
    }
}


simplex* firstBdElem( simplex* edge )
{
    return *edge->bdBegin();
}


simplex* secondBdElem( simplex* edge )
{
    simplex::BdIterator bd = edge->bdBegin();
    ++bd;
    return (*bd);
}


simplex* theOtherVertex( simplex* edge , simplex* ver )
{
     simplex::BdIterator bd = edge->bdBegin();
    if ( *bd != ver )return *bd;
    bd++;
    return *bd;
}

std::vector<simplex * > intersect( std::vector<simplex * >& f , std::vector<simplex * >& s ,unsigned a, unsigned b)
{
    std::vector<simplex * > result;
    for ( size_t fir = 0 ; fir != f.size() ; ++fir )
    {
        bool is = false;
        for ( size_t sec = 0 ; sec != s.size() ; ++sec )
        {
            if ( f[fir] == s[sec] )
            {
                is = true; break;
            }
        }
        if ( is )
        {
            if ( (f[fir]->numberInCmplx() > a) && (f[fir]->numberInCmplx() > b) )
            {
                result.push_back( f[fir] );
            }
        }
    }
    return result;
}


std::vector<simplex * > intersect( std::vector<simplex * >& f , std::vector<simplex * >& s )
{
    std::vector<simplex * > result;
    for ( size_t fir = 0 ; fir != f.size() ; ++fir )
    {
        bool is = false;
        for ( size_t sec = 0 ; sec != s.size() ; ++sec )
        {
            if ( f[fir] == s[sec] )
            {
                is = true; break;
            }
        }
        if ( is )
        {
            result.push_back( f[fir] );
        }
    }
    return result;
}



const int frameDebug = 0;
void simplicialComplex::frame( sparseSquareBinaryMatrix& mat )
{
     //creating zero dimensional simplices:

      std::vector< simplex* > zerodimensionalSimlexes( mat.size() );
      //std::list<vertex, std::allocator >::iterator it;


      //creating all vertices. This is a dummy loop which just create structure, but do not compute any boundry/cobondary
      for (  size_t i = 0 ; i != mat.size() ; ++i )
      {
         simplex * vert = new simplex;
         vert->numberInComplex = i+1;
         vert->verticesInThisSimplex.push_back( vert );
         zerodimensionalSimlexes[i] = vert;
      }
      int nrSim = mat.size()+1;


     std::vector< simplex* > onedimensionalSimlexes;
     //for every collumn
     for ( size_t i = 0 ; i != mat.size() ; ++i )
     {
         //for every nonzero element in the collumn
         std::set<size_t> column = mat(i);
         for ( std::set<size_t>::iterator it = column.begin() ; it != column.end() ; ++it )
         {
             if ( i > *it )continue;

             simplex * edg = new simplex;

             // this check is only to make sure edges are stored with vertices in increasing order
             if ( i < *it )
             {
                edg->bound.push_back( zerodimensionalSimlexes[i] );
                edg->bound.push_back( zerodimensionalSimlexes[*it] );
                edg->verticesInThisSimplex.push_back( zerodimensionalSimlexes[i] );
                edg->verticesInThisSimplex.push_back( zerodimensionalSimlexes[*it] );
             }
             else
             {
                 edg->bound.push_back( zerodimensionalSimlexes[*it] );
                 edg->bound.push_back( zerodimensionalSimlexes[i] );
                 edg->verticesInThisSimplex.push_back( zerodimensionalSimlexes[*it] );
                 edg->verticesInThisSimplex.push_back( zerodimensionalSimlexes[i] );
             }

             edg->numberInComplex = nrSim;
             ++nrSim;
             zerodimensionalSimlexes[i]->coBound.push_back(edg);
             zerodimensionalSimlexes[*it]->coBound.push_back(edg);
             onedimensionalSimlexes.push_back( edg );
         }
     }


      //now, on order to create higher dimensional simplices, in every edges E, we keep the vertives which are connected to both endpoints of E. But, to have this,
      //we first take every vertex V and fill-in the li     st of all the neighboring vertices with lower index then the index of V.
      for ( size_t i = 0 ; i != zerodimensionalSimlexes.size() ; ++i )
      {
          for ( simplex::CbdIterator cbd = zerodimensionalSimlexes[i]->cbdBegin() ; cbd != zerodimensionalSimlexes[i]->cbdEnd() ; ++cbd )
          {
              simplex* other = theOtherVertex(*cbd,zerodimensionalSimlexes[i]);
              if ( i < other->numberInCmplx() )
              {
                  zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.push_back( other );
              }
          }
          //this part of the code is not needed for the symmetrica case, but it IS for a directed one (since then, the repetitions may occuces, since there may be two edges between pair of simplices).
          //std::sort( zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.begin() , zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.end() , comparision );
          //zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.erase( unique( zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.begin(), zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.end() ), zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.end() );
      }

      //now, for every edge E, we compute the neighbors of E, which are, from definition, intersection of neighbors of the endpoits of E.
      for ( size_t i = 0 ; i != onedimensionalSimlexes.size() ; ++i )
      {
          simplex* vert1 = onedimensionalSimlexes[i]->verticesInThisSimplex[0];
          simplex* vert2 = onedimensionalSimlexes[i]->verticesInThisSimplex[1];
          onedimensionalSimlexes[i]->elementsNotFurtherThanEpsilon = intersect( vert1->elementsNotFurtherThanEpsilon ,  vert2->elementsNotFurtherThanEpsilon );
      }
     this->elements.push_back(zerodimensionalSimlexes);
     this->elements.push_back(onedimensionalSimlexes);

     std::cerr << "In dimension 0 we have " << zerodimensionalSimlexes.size() << " simplices \n";
     std::cerr << "In dimension 1 we have " << onedimensionalSimlexes.size() << " simplices \n";

     this->numberOfSimplexes = nrSim;
}//frame


int simplicialComplex::createHigherdimensionalSimplexes()
{
    int numberInAComplex = this->numberOfSimplexes;
    std::vector< simplex* > listOfCreatedSimplexes;
    int currentGradationLength = this->elements.size();
    int numberOfCreatedSimplexes = 0;

     std::vector< simplex* >::iterator it1, ia;
     //for every simplex in the currently maximal dimensions in the complex:
    for ( it1 = this->elements[currentGradationLength-1].begin() ; it1 != this->elements[currentGradationLength-1].end() ; ++it1 )
    {
          for ( unsigned int neighIterator = 0 ; neighIterator != (*it1)->elementsNotFurtherThanEpsilon.size() ; ++neighIterator)
          {
             //creation of a new simplex based on vertices of *it1 and the vertex given by *neighIterator
             simplex * sim = new simplex;
             for ( unsigned int ka = 0 ; ka < (*it1)->verticesInThisSimplex.size() ; ++ka )
             {
                 sim->verticesInThisSimplex.push_back( (*it1)->verticesInThisSimplex [ka] );
             }
             sim->verticesInThisSimplex.push_back( (*it1)->elementsNotFurtherThanEpsilon[neighIterator]  );




             //std::setting boundary and coboundary:
             for ( simplex::BdIterator bd = (*it1)->bdBegin() ; bd != (*it1)->bdEnd() ; ++bd )
             {
                 for ( simplex::CbdIterator cbd = (*bd)->cbdBegin() ; cbd != (*bd)->cbdEnd() ; ++cbd )
                 {
                     if ( (*cbd) == (*it1) )continue;
                     bool isThere = false;
                     //we are checking if (*it1)->elementsNotFurtherThanEpsilon[neighIterator] is one of the vertices of this simplex.
                     for ( size_t rr = 0 ; rr != (*cbd)->verticesInThisSimplex.size() ; ++rr )
                     {
                         if ( (*cbd)->verticesInThisSimplex[rr] == (*it1)->elementsNotFurtherThanEpsilon[neighIterator] )
                         {
                             isThere = true;
                             break;
                         }
                     }
                     if ( isThere )
                     {
                         (*cbd)->coBound.push_back( sim );
                         sim->bound.push_back( *cbd );
                         break;
                     }
                 }
             }
             (*it1)->coBound.push_back(sim);
             sim->bound.push_back((*it1));


             sim->numberInComplex = numberInAComplex;
             numberInAComplex++;
             sim->intersection( (*it1), (*it1)->elementsNotFurtherThanEpsilon[neighIterator] );
             numberOfCreatedSimplexes++;
             listOfCreatedSimplexes.push_back(sim);
          }
    }//for it1
    if ( listOfCreatedSimplexes.size() )
    {
        this->numberOfSimplexes = numberInAComplex;
        this->elements.push_back( listOfCreatedSimplexes );
    }
    std::cerr << "In dimension " << this->elements.size() << " we have : " << listOfCreatedSimplexes.size() << " simplices \n";
    return numberOfCreatedSimplexes;
}//createHigherdimensionalSimplexes





simplicialComplex::simplicialComplex()
{
     this->disposed = false;
}



const int cechCoWallAlgDebug = 0;
simplicialComplex::simplicialComplex( sparseSquareBinaryMatrix& mat , int chopDimnesion  )
{
    this->disposed = false;
    //cout << "Begin creation of a graph \n";
    this->frame(mat);
    int dimension = 3;
    int numberOfCreatedSimplex = 3;

    while ( true )
    {
          //cerr << "Begin creation of higher dimensional simplices \n";
          numberOfCreatedSimplex = this->createHigherdimensionalSimplexes();
          if ( numberOfCreatedSimplex == 0 )break;
          if ( (chopDimnesion != -1)&&(dimension >= chopDimnesion) )break;
          ++dimension;
     }
}

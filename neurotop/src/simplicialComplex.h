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


#include <list>
#include <sstream>
#include <algorithm>


/*
* This is a directed simplicial complex class. A directed simplicial complex is a collection of sequences of simplices of different dimensions having the property that every subsequence of (vertices) of a simplex
* that belongs to an abstract simplicial complex is a simplex in that complex.
*/
class simplicialComplex
{
public:
       /*
       * Constructor of an directed simplicial complex that takes as an input a sparseSquareBinaryMatrix. The second optional parameter is a maximal dimension of simplices we want to create.
       */
       simplicialComplex( sparseSquareBinaryMatrix& mat , int BORDER = -1 );

       /*
       * Constructor of an directed simplicial complex that creates empty directed simplicial complex.
       */
       simplicialComplex();



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
          for ( unsigned dim = 0 ; dim != this->elements.size() ; ++dim )
          {
              //std::cerr <<"dim : " << dim << " ilosc : " << this->elements[dim].size() << std::endl;
              for (  std::vector< simplex* >::iterator it = this->elements[dim].begin() ; it != this->elements[dim].end() ; ++it )
              {
                  if ( (*it)->delet )continue;
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
              if ( iloscELementowWBrzegu == 1 )
              {
                  ++ilosc;
                  top->delet = true;
                  second->delet = true;
                  for (  simplex::BdIterator bd = top->bdBegin() ; bd != top->bdEnd() ; ++bd )
                  {
                      freeFaces.push_back( *bd );
                  }
                  for (  simplex::BdIterator bd = second->bdBegin() ; bd != second->bdEnd() ; ++bd )
                  {
                      freeFaces.push_back( *bd );
                  }
              }
          }
          std::cerr << "Number of reduced pairs : " << ilosc << std::endl;
          return ilosc;
      }


       /*
       * This procedure performs so called coreductions, see
       * http://link.springer.com/article/10.1007%2Fs00454-008-9073-y
       * It remove some cells of the complex keeping its reduced homology unchanged. In particular, all homology of dimension greater than zero of the complex will not be affected by this procedure.
       */
      int coreductions()
      {
          std::cerr << "Coreductions \n";
          //szukanie i usuwanie pierwszego elementu:
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
          std::list<simplex*> freeFaces;
          for ( unsigned dim = 0 ; dim != this->elements.size() ; ++dim )
          {
              //std::cerr <<"dim : " << dim << " ilosc : " << this->elements[dim].size() << std::endl;
              for (  std::vector< simplex* >::iterator it = this->elements[dim].begin() ; it != this->elements[dim].end() ; ++it )
              {
                  if ( (*it)->delet )continue;
                  int iloscELementowWKobrzegu = 0;

                  for (  std::vector<simplex*>::iterator bd = (*it)->bound.begin() ; bd != (*it)->bound.end() ; ++bd )
                  {
                      if ( !(*bd)->delet )++iloscELementowWKobrzegu;
                  }
                  if ( iloscELementowWKobrzegu == 1 )
                  {
                      freeFaces.push_back( *it );
                  }
              }
          }


        //std::cerr << "freeFaces.size() " << freeFaces.size() << std::endl;

          while ( !freeFaces.empty() )
          {
              simplex* top = *freeFaces.begin();
              freeFaces.pop_front();
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
                  top->delet = true;
                  second->delet = true;
                  for (  simplex::CbdIterator cbd = top->cbdBegin() ; cbd != top->cbdEnd() ; ++cbd )
                  {
                      freeFaces.push_back( *cbd );
                  }
                  for (  simplex::CbdIterator cbd = second->cbdBegin() ; cbd != second->cbdEnd() ; ++cbd )
                  {
                      freeFaces.push_back( *cbd );
                  }
              }
          }
          std::cerr << "Number of coreduced pairs : " << ilosc << std::endl;\
          return ilosc;
      }

       /*
       * Destructor of simplicial complex class.
       */
      ~simplicialComplex()
      {
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
       * A method that write the complex to a file. Writing is done in the same order as in << operator.
       */
      void writeToFile(char* filename);

      /*
       * Accessor to the list of simplices graded by dimension.
      */
      std::vector< std::vector< simplex* > >& elemen(){return this->elements;}

      friend std::vector< size_t > computeHomology( simplicialComplex* cmplx );

protected:

       /*
       * A method that is called by a constructor of simplicialComplex class. It create a  2 skeleton of a complex (vertices, edges and 2-siplices/triangles) from a connection matrix.
       */
       void frame( sparseSquareBinaryMatrix& mat );

       /*
       * A method that is called by a constructor of simplicialComplex class. When the frame() method is done, it create the higher dimensional simplices. It create simplices of
       * dimension 1 higher than the dimension of simplices that are currently present in the complex,
       */
       int createHigherdimensionalSimplexes();

       bool deletedd;
       std::vector< std::vector< simplex* > > elements;
       unsigned int numberOfSimplexes;
       bool disposed;
};//simplicialComplex





void simplicialComplex::writeMaximalSimplicesToFile( char* filename )
{
    for ( size_t dim = 0 ; dim != this->elements.size() ; ++dim )
    {
        std::ofstream out;
        std::ostringstream name;
        name << filename << "directed_maxSimpl_dim_" << dim << ".txt";
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
            if ( (f[fir]->numberInComplex > a) && (f[fir]->numberInComplex > b) )
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


bool comparision( simplex* s1, simplex* s2 )
{
    return ( s1->numberInCmplx() < s2->numberInCmplx() );
}


//this procedure et sup the verticesInThisSimplex elements so that they appear in the correct order.
void setUpVerticesInThisSimplex( simplex* tri , std::vector< simplex* > facesOfThisSimplex )
{
    bool dbg = false;

    //the key observation here is that all the elements from the std::vector facesOfThisSimplex but one will have the same element at the beginning,
    //which will be the lowest element in the created simplex.
    if ( facesOfThisSimplex.size() < 3 )return;//this should not happened.

    simplex* first = facesOfThisSimplex[0]->verticesInThisSimplex[0];
    simplex* second = facesOfThisSimplex[1]->verticesInThisSimplex[0];
    simplex* third = facesOfThisSimplex[2]->verticesInThisSimplex[0];

    simplex* theLowestVertexInThisSimplex = 0;

    if ( first == second )theLowestVertexInThisSimplex = first;
    if ( first == third )theLowestVertexInThisSimplex = first;
    if ( second == third )theLowestVertexInThisSimplex = second;

    //now, when we know which is the lowest simplex, it is time to find a face among facesOfThisSimplex that do not contain it:
    int theOddOne = -1;
    for ( size_t i = 0 ; i != facesOfThisSimplex.size() ; ++i )
    {
        if ( facesOfThisSimplex[i]->verticesInThisSimplex[0] != theLowestVertexInThisSimplex )
        {
            theOddOne = i;
            break;
        }
    }

    if ( theOddOne == -1 )
    {
        std::cerr << "Error, this should not happened \n";

        for ( size_t i = 0 ; i != facesOfThisSimplex.size() ; ++i )
        {
            std::cerr << "facesOfThisSimplex[i] : " << *facesOfThisSimplex[i] << std::endl;
        }
        getchar();

        return;
    }

    tri->verticesInThisSimplex.push_back( theLowestVertexInThisSimplex );
    for ( size_t i = 0 ; i != facesOfThisSimplex[theOddOne]->verticesInThisSimplex.size() ; ++i )
    {
        tri->verticesInThisSimplex.push_back( facesOfThisSimplex[theOddOne]->verticesInThisSimplex[i] );
    }

    if ( dbg )
    //if ( tri->dim()  )
    {
        std::cerr << "We have created a simplex : " << *tri << " based on : \n";
        for ( size_t aa = 0 ; aa != facesOfThisSimplex.size() ; ++aa )
        {
            std::cerr << *facesOfThisSimplex[aa] << std::endl;
        }
        getchar();
    }

}//setUpVerticesInThisSimplex


const int frameDebug = 0;
void simplicialComplex::frame( sparseSquareBinaryMatrix& mat )
{
      bool dbg = false;
      //creating zero dimensional simplices:

      std::vector< simplex* > zerodimensionalSimlexes(mat.size());
      //std::list<vertex, std::allocator >::iterator it;

      if (dbg)
      {
          std::cerr << "Starting to create zero dimensional simplices \n";
          std::cerr << "mat.size : " << mat.size() << std::endl;
      }

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
     for ( size_t i = 0 ; i < mat.size() ; ++i )
     {
         //for every nonzero element in the collumn
         std::set<size_t> column = mat(i);
         for ( std::set<size_t>::iterator it = column.begin() ; it != column.end() ; ++it )
         {
             //there should be a directed edge from i to *it!
             simplex * edg = new simplex;
             edg->bound.push_back( zerodimensionalSimlexes[i] );
             edg->bound.push_back( zerodimensionalSimlexes[*it] );

             //now this ordering matters.
             edg->verticesInThisSimplex.push_back( zerodimensionalSimlexes[i] );
             edg->verticesInThisSimplex.push_back( zerodimensionalSimlexes[*it] );

             edg->numberInComplex = nrSim;
             ++nrSim;
             zerodimensionalSimlexes[i]->coBound.push_back(edg);
             zerodimensionalSimlexes[*it]->coBound.push_back(edg);
             onedimensionalSimlexes.push_back( edg );
         }
     }

      if (dbg)std::cerr <<"Starting to create 1-dimensional simplices \n";

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
          //either way of removing duplicates seems to be OK.

          std::sort( zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.begin() , zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.end() , comparision );
          std::vector<simplex*>::iterator ita = std::unique (zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.begin(), zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.end() , same );
          zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.resize( std::distance(zerodimensionalSimlexes[i]->elementsNotFurtherThanEpsilon.begin(),ita) );
      }

      for ( size_t i = 0 ; i != onedimensionalSimlexes.size() ; ++i )
      {
          simplex* vert1 = onedimensionalSimlexes[i]->verticesInThisSimplex[0];
          simplex* vert2 = onedimensionalSimlexes[i]->verticesInThisSimplex[1];
          onedimensionalSimlexes[i]->elementsNotFurtherThanEpsilon = intersect( vert1->elementsNotFurtherThanEpsilon ,  vert2->elementsNotFurtherThanEpsilon );
      }


     std::vector< simplex* > twodimensionalSimlexes;

    //here we are creating two simplices. The trick in this type of complex is that there may exist two different 2-simplices supported in the same set of vertices.
    //We achieve this by searching cycles of the length 3 in the complex.
    //To do so, we iterate from first to the last vertex. For each of them we check if the initial vertex has the lowest lable in the complex.
    //Moreover, we assume, that if we have vertices v1, v2, v3 in order as they appear in the cycle, then v1 < v2 < v3. This simple idea
    //quarantees that every cycle is considered only once. Every such a cycle is a candidate for a 2 simplex, but due to the orientation not every cycle will give rise to a simplex.

    std::vector< std::vector<simplex*> > candidatesForTriangels;



    //#pragma omp parallel for
    for ( size_t i = 0 ; i < zerodimensionalSimlexes.size() ; ++i )
    {
        if (dbg){std::cerr << "i : " << i << std::endl;}
        //we are searching here for all cycles of a length 3 started in zerodimensionalSimlexes[i]
        simplex* vert1 = zerodimensionalSimlexes[i];
        for (  simplex::CbdIterator edge1 = vert1->cbdBegin() ; edge1 != vert1->cbdEnd() ; ++edge1 )
        {
            simplex* vert2 = theOtherVertex( *edge1 , vert1 );
            if ( vert2->numberInCmplx() < vert1->numberInCmplx() )continue;
            for (  simplex::CbdIterator edge2 = vert2->cbdBegin() ; edge2 != vert2->cbdEnd() ; ++edge2 )
            {
                if ( *edge1 == *edge2 )continue;
                simplex* vert3 = theOtherVertex(*edge2,vert2);
                if ( vert3->numberInCmplx() < vert2->numberInCmplx() )continue;
                for (  simplex::CbdIterator edge3 = vert3->cbdBegin() ; edge3 != vert3->cbdEnd() ; ++edge3 )
                {
                    if ( *edge3 == *edge2 )continue;
                    simplex* vert4 = theOtherVertex(*edge3,vert3);
                    if (vert4 == vert1)
                    {
                        //we found a cycle! is yhe lowest vertex in this cycle.
                        std::vector<simplex*> candidate(6);
                        candidate[0] = *edge1;
                        candidate[1] = *edge2;
                        candidate[2] = *edge3;
                        candidate[3] = vert1;
                        candidate[4] = vert2;
                        candidate[5] = vert3;
                        //std::cerr << **edge1 << " , " << **edge2 << " , " << **edge3 << std::endl;
                        //getchar();
                        candidatesForTriangels.push_back( candidate );
                    }
                }
            }
        }
    }//for ( size_t i = 0 ; i != zerodimensionalSimlexes.size() ; ++i )
    if (dbg)
    {
        std::cerr << "candidatesForTriangels.size() : " << candidatesForTriangels.size() << "\n";
    }




    //#pragma omp parallel for
    for ( size_t i = 0 ; i < candidatesForTriangels.size() ; ++i  )
    {
          simplex* edg12 = candidatesForTriangels[i][0];
          simplex* edg13 = candidatesForTriangels[i][2];
          simplex* edg23 = candidatesForTriangels[i][1];

          simplex* vert1 = candidatesForTriangels[i][3];
          simplex* vert2 = candidatesForTriangels[i][4];
          simplex* vert3 = candidatesForTriangels[i][5];

          if ( dbg )
          {
              std::cerr << "edg12 : " << *edg12 << "\n";
              std::cerr << "edg13 : " << *edg13 << "\n";
              std::cerr << "edg23 : " << *edg23 << "\n";

              std::cerr << "vert1 : " << *vert1 << "\n";
              std::cerr << "vert2 : " << *vert2 << "\n";
              std::cerr << "vert3 : " << *vert3 << "\n";
              getchar();
          }


          bool doWeAddTriangle = true;


          //jezeli orientacja tworzy cykl, to nie dodajemy trojkata:
          if (
                 (
                    ( secondBdElem(edg12) == firstBdElem(edg23) )  //clockwise
                    &&
                    ( secondBdElem(edg23) == firstBdElem(edg13) )
                    &&
                    ( secondBdElem(edg13) == firstBdElem(edg12) )
                 )
                 ||
                 (
                    ( firstBdElem(edg12) == secondBdElem(edg23) )  //counter clockwise
                    &&
                    ( firstBdElem(edg23) == secondBdElem(edg13) )
                    &&
                    ( firstBdElem(edg13) == secondBdElem(edg12) )
                 )
             )
          {
             doWeAddTriangle = false;
          }

          if ( doWeAddTriangle )
          {
              if (dbg)
              {
                std::cerr << "firstBdElem(edg12) : " << *firstBdElem(edg12) << "\n";
                std::cerr << "secondBdElem(edg12) : " << *secondBdElem(edg12) << "\n";
                std::cerr << "firstBdElem(edg23) : " << *firstBdElem(edg23) << "\n";
                std::cerr << "secondBdElem(edg23) : " << *secondBdElem(edg23) << "\n";
                std::cerr << "firstBdElem(edg13) : " << *firstBdElem(edg13) << "\n";
                std::cerr << "secondBdElem(edg13) : " << *secondBdElem(edg13) << "\n";

                std::cerr << *edg12 << " , " << *edg23 << " , " << *edg13 << std::endl;
                std::cerr<< "Dodajemy trojkat, uzpelniamy bebechy";

              }

              simplex * tri = new simplex;
              twodimensionalSimlexes.push_back(tri);
              tri->bound.push_back(edg13);
              tri->bound.push_back(edg12);
              tri->bound.push_back(edg23);

              edg13->coBound.push_back(tri);
              edg12->coBound.push_back(tri);
              edg23->coBound.push_back(tri);
              tri->numberInComplex = nrSim;
              ++nrSim;

              std::vector< simplex* > facesOfThisSimplex(3);
              facesOfThisSimplex[0] = edg12;
              facesOfThisSimplex[1] = edg13;
              facesOfThisSimplex[2] = edg23;
              setUpVerticesInThisSimplex( tri , facesOfThisSimplex );


              //for the time of changes in the code, we comment out this part:
              //tri->verticesInThisSimplex.push_back(vert1);
              //tri->verticesInThisSimplex.push_back(vert2);
              //tri->verticesInThisSimplex.push_back(vert3);

              std::vector<simplex * > inte = intersect( vert1->elementsNotFurtherThanEpsilon , vert2->elementsNotFurtherThanEpsilon );
              std::vector<simplex * > intersection = intersect( inte , vert3->elementsNotFurtherThanEpsilon );
              tri->elementsNotFurtherThanEpsilon = intersection;
          }
    }
     this->elements.push_back(zerodimensionalSimlexes);
     this->elements.push_back(onedimensionalSimlexes);
     this->elements.push_back(twodimensionalSimlexes);

     std::cerr << "In dimension 0 we have " << zerodimensionalSimlexes.size() << " simplices \n";
     std::cerr << "In dimension 1 we have " << onedimensionalSimlexes.size() << " simplices \n";
     std::cerr << "In dimension 2 we have " << twodimensionalSimlexes.size() << " simplices \n";

     this->numberOfSimplexes = nrSim;
}//frame


//CAUTION! - this procedure is different with respect to the one used for the case when direction is not taken into account!
int simplicialComplex::createHigherdimensionalSimplexes()
{
    int numberInAComplex = this->numberOfSimplexes;
    std::vector< simplex* > listOfCreatedSimplexes;
    int currentGradationLength = this->elements.size();
    int numberOfCreatedSimplexes = 0;

    std::vector< simplex* >::iterator it1, ia;
    //for every element of the top dimension in the complex (top dimension before calling this procedure).
    int aad = 0 ;
    for ( it1 = this->elements[currentGradationLength-1].begin() ; it1 != this->elements[currentGradationLength-1].end() ; ++it1 )
    {
          ++aad;
          //and for every vertex which is a neighbour of the element:
          for ( unsigned int neighIterator = 0 ; neighIterator != (*it1)->elementsNotFurtherThanEpsilon.size() ; ++neighIterator)
          {
                 //I want to create a simplex based on the *it1 and the vertex (*it1)->elementsNotFurtherThanEpsilon[neighIterator]
                 //But there can be a lot of such a simplices. Or there can be no such a simplices at all. We are checking it in here.
                 //In order to do it, we check the boundary of *it1. Take every element *bd in the boundary of *it1.
                 //Having *bd we are checking all elements *cbd in the coboundary of *bd (except from *it1).
                 //If *cbd contains a vertex (*it1)->elementsNotFurtherThanEpsilon[neighIterator], then
                 std::vector< std::vector<simplex*> > boundaryElemsOfNewSimplex;
                 for ( simplex::BdIterator bd = (*it1)->bdBegin() ; bd != (*it1)->bdEnd() ; ++bd )
                 {
                       std::vector< simplex* > elementsToAddToBdInCoboundayofBD;
                       for ( simplex::CbdIterator cbd = (*bd)->cbdBegin() ; cbd != (*bd)->cbdEnd() ; ++cbd )
                       {
                            if ( *cbd == *it1 )continue;
                            //if *cbd contains a vertex (*it1)->elementsNotFurtherThanEpsilon[neighIterator]
                            bool doesCbdContainThisVertex = false;
                            for ( size_t i = 0 ; i != (*cbd)->verticesInThisSimplex.size() ; ++i )
                            {
                                if ( (*cbd)->verticesInThisSimplex[i] == (*it1)->elementsNotFurtherThanEpsilon[neighIterator] )
                                {
                                    doesCbdContainThisVertex = true;
                                    break;
                                }
                            }
                            if ( doesCbdContainThisVertex )
                            {
                                //then *cbd should be considered in the boundary of the newly created simplex
                                elementsToAddToBdInCoboundayofBD.push_back( *cbd );
                            }
                       }
                       if ( elementsToAddToBdInCoboundayofBD.size() == 0 )
                       {
                           //Yes, this can happend! Suppose that a simplex *it1 has a neighbor vertex aa. because of the orientation hovewer we do not know if all the
                           //faces of *it1 has connection with aa. In this case, we do not create higher dimensional simplices based on it1. Therefore, we clear the
                           //boundaryElemsOfNewSimplex vectr and break the for ( simplex::BdIterator bd = (*it1)->bdBegin() ; bd != (*it1)->bdEnd() ; ++bd ) loop.
                           boundaryElemsOfNewSimplex.clear();
                           break;
                       }

                       //Now imagine (*it) is a triangle [a,b,c] and suppose that the neighbor element is a vertex d. Suppose moreover, that we have
                       //two copies of each of the following triangles: [a,b,d], [a,c,d], [b,c,d]. Suppose that the order of boundary elements of a triangle
                       //[a,b,c] is [a,b], [a,c], [b,c].
                       //Then, we have first (*bd) = [a,b] and in the loop above we find two copies of [a,b,d] and add them to elementsToAddToBdInCoboundayofBD.
                       //Since this is a first iteration, the list elementsToAddToBdInCoboundayofBD became boundaryElemsOfNewSimplex. But, we need do be a bit careful here.
                       //elementsToAddToBdInCoboundayofBD is std::vector< simplex* >, while boundaryElemsOfNewSimplex is a std::vector< std::vector<simplex*> >.
                       //In this case, every element of elementsToAddToBdInCoboundayofBD became one element std::vector of boundaryElemsOfNewSimplex.
                       //Then the second iteration comes, and (*bd) = [a,c]. Again, in the for loop in above we find both triangles [a,c,d]. They are in elementsToAddToBdInCoboundayofBD list.
                       //Then, the size of boundaryElemsOfNewSimplex would double, since for everything that was there previously, we can choose either of triangles
                       //[a,c,d] to continue. This is what is done in the for loop in below:


                       std::vector< std::vector<simplex*> > boundaryElemsOfNewSimplexNew;
                       for ( size_t aa = 0 ; aa != elementsToAddToBdInCoboundayofBD.size() ; ++aa )
                       {
                           if ( boundaryElemsOfNewSimplex.size() )
                           {
                               for ( size_t bb = 0 ; bb != boundaryElemsOfNewSimplex.size() ; ++bb )
                               {
                                   std::vector<simplex*> newBoundarySlot;
                                   for ( size_t cc = 0 ; cc != boundaryElemsOfNewSimplex[bb].size() ; ++cc )
                                   {
                                       newBoundarySlot.push_back( boundaryElemsOfNewSimplex[bb][cc] );
                                   }
                                   newBoundarySlot.push_back(elementsToAddToBdInCoboundayofBD[aa]);
                                   boundaryElemsOfNewSimplexNew.push_back( newBoundarySlot );
                               }
                           }
                           else
                           {
                               //boundaryElemsOfNewSimplex.size() == 0
                               std::vector<simplex*> newBoundarySlot;
                               newBoundarySlot.push_back(elementsToAddToBdInCoboundayofBD[aa]);
                               boundaryElemsOfNewSimplexNew.push_back( newBoundarySlot );
                           }
                       }
                       boundaryElemsOfNewSimplex = boundaryElemsOfNewSimplexNew;
                 }


                //right now we have the whole list boundaryElemsOfNewSimplex, which is a std::vector< std::vector<simplex*> >. Now, based on this, we will create new simplices:
                for ( size_t simplNr = 0 ; simplNr != boundaryElemsOfNewSimplex.size() ; ++simplNr )
                {
                     //Now, here comes the tricky part. For a simplex *it1 and its neighbor (*it1)->elementsNotFurtherThanEpsilon[neighIterator] we have std::vector
                     //of std::vectors of simplices. Each of the std::vectors of simplices spins around boundary of *it1, and all of them contains a vertex
                     //(*it1)->elementsNotFurtherThanEpsilon[neighIterator]. But, this is not yet implies, that we can create a new simplex based on every such a std::vector.
                     //The reason is because we do not know if any two elements of a std::vector (as well as the first and last element of this std::vector)
                     //share a simplex of codimension one. If they all do, we are OK, and we can create a simplex. But, if they do not, we cannot do it.
                     //In below we have a test used to check it. We check only elements from boundaryElemsOfNewSimplex std::vector, since the intersection of *it1 wit every
                     //element of boundaryElemsOfNewSimplex has already been check when creating the std::vector boundaryElemsOfNewSimplex.
                     bool canWeMakeASimplex = true;
                     for ( size_t fir = 0 ; fir != boundaryElemsOfNewSimplex[simplNr].size() ; ++fir )
                     {
                         for ( size_t sec = 0 ; sec != boundaryElemsOfNewSimplex[simplNr].size() ; ++sec )
                         {
                              if ( fir == sec )continue;
                              //we are checking if boundaryElemsOfNewSimplex[simplNr][fir] and boundaryElemsOfNewSimplex[simplNr][sec] have something in common in boundary:
                              bool isThere = false;
                              for ( simplex::BdIterator bdFir = boundaryElemsOfNewSimplex[simplNr][fir]->bdBegin() ; bdFir != boundaryElemsOfNewSimplex[simplNr][fir]->bdEnd() ; ++bdFir )
                              {
                                  for ( simplex::BdIterator bdSec = boundaryElemsOfNewSimplex[simplNr][sec]->bdBegin() ; bdSec != boundaryElemsOfNewSimplex[simplNr][sec]->bdEnd() ; ++bdSec )
                                  {
                                      if ( *bdFir == *bdSec )
                                      {
                                          isThere = true;
                                          break;
                                      }
                                  }
                                  if ( isThere )
                                  {
                                      break;
                                  }
                              }
                              if ( !isThere )
                              {
                                  //in this case we were not able to find an element which is a codimension one face between boundaryElemsOfNewSimplex[simplNr][fir] and boundaryElemsOfNewSimplex[simplNr][sec]
                                  canWeMakeASimplex = false;
                                  break;
                              }
                         }
                         if ( !canWeMakeASimplex )break;
                     }

                     if ( canWeMakeASimplex )
                     {
                         //We are creating a new simplex:
                         simplex * sim = new simplex;

                         std::vector<unsigned> numbersOfVerticesInSimplex;
                         for ( unsigned int ka = 0 ; ka < (*it1)->verticesInThisSimplex.size() ; ++ka )
                         {
                             numbersOfVerticesInSimplex.push_back( (*it1)->verticesInThisSimplex[ka]->numberInCmplx() );
                         }
                         numbersOfVerticesInSimplex.push_back( (*it1)->elementsNotFurtherThanEpsilon[neighIterator]->numberInCmplx() );
                         std::sort( numbersOfVerticesInSimplex.begin() , numbersOfVerticesInSimplex.end() ); //we do not need to sort it here in fact, but just to find
                                                                                                             //min and max. This part of the code should be optimized!


                        boundaryElemsOfNewSimplex[simplNr].push_back( *it1 );
                        setUpVerticesInThisSimplex( sim , boundaryElemsOfNewSimplex[simplNr] );




                         //setting up boundary and coboundary:
                         for ( size_t aa = 0 ; aa != boundaryElemsOfNewSimplex[simplNr].size() ; ++aa )
                         {
                              boundaryElemsOfNewSimplex[simplNr][aa]->coBound.push_back(sim);
                              sim->bound.push_back( boundaryElemsOfNewSimplex[simplNr][aa] );
                         }


                         //and the remaining parameters in the simplex data struture:
                         sim->numberInComplex = numberInAComplex++;
                         sim->intersection( (*it1), (*it1)->elementsNotFurtherThanEpsilon[neighIterator] );
                         numberOfCreatedSimplexes++;
                         listOfCreatedSimplexes.push_back(sim);
                     }
                }
          }//for ( unsigned int neighIterator
    }//for it1
    if ( listOfCreatedSimplexes.size() )
    {
        this->numberOfSimplexes = numberInAComplex;
        this->elements.push_back( listOfCreatedSimplexes );
    }
    return numberOfCreatedSimplexes;
}//createHigherdimensionalSimplexes





simplicialComplex::simplicialComplex()
{
     this->disposed = false;
}



simplicialComplex::simplicialComplex( sparseSquareBinaryMatrix& mat , int BORDER )
{
    this->disposed = false;
    this->frame(mat);
    int dimension = 3;
    int numberOfCreatedSimplex = 3;

    unsigned int timeToEnd = 1;

    if ( BORDER == 1 ){timeToEnd=0;}
    while ( timeToEnd != 0 )
    {
          //std::cerr << "Creating simplices of dimension : " << dimension << std::endl;
          numberOfCreatedSimplex = this->createHigherdimensionalSimplexes();
          std::cerr << "In dimension " << dimension << " we have " << numberOfCreatedSimplex << " simplices \n";;

          ++dimension;
          if ( BORDER != -1 )
          {
              if ( BORDER <= dimension ) break;
          }
          if ( numberOfCreatedSimplex == 0 )break;
    }
}

std::vector< int > howManyMultipleTriangelsThereAre( simplicialComplex * cmplx )
{
    std::vector< std::vector< int* > > counter( cmplx->elemen()[0].size() );
    // counter[i] -- set of all triangels with i as a smallest vertex
    for ( std::vector< simplex* >::iterator it = cmplx->elemen()[2].begin() ; it != cmplx->elemen()[2].end() ; ++it )
    {
        //let us take vertices in the somples
        std::vector<int> vert(3);
        vert[0] = (*it)->verticesInThisSimplex[0]->numberInComplex;
        vert[1] = (*it)->verticesInThisSimplex[1]->numberInComplex;
        vert[2] = (*it)->verticesInThisSimplex[2]->numberInComplex;
        std::sort( vert.begin() , vert.end() );

        //checing in counter[ vert[0] ]if there is a pair ( vert[1] , vert[2] , ?? )
        bool isThere = false;
        for ( size_t i = 0 ; i != counter[ vert[0] ].size() ; ++i )
        {
            if ( (counter[ vert[0] ][i][0] == vert[1]) && (counter[ vert[0] ][i][1] == vert[2]) )
            {
                ++(counter[ vert[0] ][i][2]);
                isThere = true;
                break;
            }
        }
        if ( !isThere )
        {   int* aa = new int[3];
            aa[0] = vert[1];
            aa[1] = vert[2];
            aa[2] = 1;
            counter[ vert[0] ].push_back( aa );
        }
    }
    int maxDeg = -1;
    for ( size_t i = 0 ; i != counter.size() ; ++i )
    {
        for ( size_t j = 0 ; j != counter[i].size() ; ++j )
        {
            if ( counter[i][j][2] > maxDeg )maxDeg = counter[i][j][2];
        }
    }

    std::vector< int > degreeOfTri(maxDeg);
    for ( int i = 0 ; i != maxDeg ; ++i )degreeOfTri[i] = 0;


    for ( size_t i = 0 ; i != counter.size() ; ++i )
    {
        for ( size_t j = 0 ; j != counter[i].size() ; ++j )
        {
            degreeOfTri[ counter[i][j][2] ]++;
        }
    }

    return degreeOfTri;
}


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

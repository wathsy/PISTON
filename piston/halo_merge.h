#ifndef HALO_MERGE_H
#define HALO_MERGE_H

#include <piston/halo.h>

// When TEST is defined, output all results
//#define TEST

namespace piston
{

class halo_merge : public halo
{
public:

  double max_ll, min_ll; // maximum & minimum linking lengths

  float totalTime;              // total time taken for halo finding
  float  cubeLen;               // length of the cube
  unsigned int mergetreeSize;   // total size of the global merge tree
  unsigned int side, size, ite; // variable needed to determine the neighborhood cubes

  unsigned int cubes;                         // total number of cubes which should be considered, usually this is equal to cubesNonEmpty
  unsigned int numOfCubes;                    // total number of cubes in space
  unsigned int cubesInX, cubesInY, cubesInZ;  // number of cubes in each dimension

  unsigned int chunks;          // amount of chunks which should be considered

  thrust::device_vector<int>   sizeOfChunks;  // size of each chunk of cubes
  thrust::device_vector<int>   startOfChunks; // start of each chunk of cubes

  unsigned int binSize, bins; // binSize & number of bins for grouping the cubes
  thrust::device_vector<int>   binStart; //start of cubeMapping id for each bin

  unsigned int numOfEdges;      // total number of edges in space

  thrust::device_vector<int>   idOriginal;           // original particle ids
  thrust::device_vector<unsigned int> cubeId;        // for each particle, cube id

  thrust::device_vector<unsigned int> cubeMapping;   // array with only nonempty cube ids
  thrust::device_vector<int>   particleSizeOfCubes;  // number of particles in cubes (only nonempty cube ids)
  thrust::device_vector<int>   particleStartOfCubes; // stratInd of cubes (only nonempty cube ids)

  thrust::device_vector<int>   edgesSrc, edgesDes;   // edge of cubes - src & des
  thrust::device_vector<float> edgesWeight;          // edge of cubes - weight
  thrust::device_vector<int>   edgeSizeOfCubes;      // size of edges in cubes
  thrust::device_vector<int>   edgeStartOfCubes;     // start of edges in cubes

  thrust::device_vector<int>   tmpIntArray;          // temporary array

  thrust::device_vector<int>   tmpNxt, tmpFree;      // stores details of free items in merge tree

  // rest of the details for leafs
  thrust::device_vector<int>   leafParent, leafParentS;

  // stores all node details of the merge tree
  thrust::device_vector<int>   nodeI;                  // index of each node
  thrust::device_vector<int>   nodeM;                  // index of each node
  thrust::device_vector<int>   nodeCount;              // size of each node
  thrust::device_vector<int>   nodeParent;             // parent of each node
  thrust::device_vector<float> nodeValue;              // function value of each node
  thrust::device_vector<float> nodeX, nodeY, nodeZ;    // positions of each node
  thrust::device_vector<float> nodeVX, nodeVY, nodeVZ; // velocities of each node



  halo_merge(float min_linkLength, float max_linkLength, std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1) : halo(filename, format, n, np, rL)
  {
    if(numOfParticles!=0)
    {
      // initializations

      struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff1, diff2, diff3, diff4;

      // un-normalize linkLengths so that it will work with box size distances
      min_ll  = min_linkLength*xscal; // get min_linkinglength
      max_ll  = max_linkLength*xscal; // get max_linkinglength
      cubeLen = min_ll / std::sqrt(3); // min_ll*min_ll = 3*cubeLen*cubeLen

      if(cubeLen <= 0) { std::cout << "--ERROR : please specify a valid cubeLen... current cubeLen is " << cubeLen/xscal << std::endl; return; }

      // get the number of neighbors which should be checked for each cube
      side = (1 + std::ceil(max_ll/cubeLen)*2);
      size = side*side*side;
      ite = (size-1)/2;

      // set total number of cubes
      cubesInX = std::ceil((uBoundX-lBoundX)/cubeLen); if (cubesInX==0) cubesInX = 1;
      cubesInY = std::ceil((uBoundY-lBoundY)/cubeLen); if (cubesInY==0) cubesInY = 1;
      cubesInZ = std::ceil((uBoundZ-lBoundZ)/cubeLen); if (cubesInZ==0) cubesInZ = 1;
      numOfCubes = cubesInX*cubesInY*cubesInZ;

      std::cout << "lBoundS " << lBoundX << " " << lBoundY << " " << lBoundZ << std::endl;
      std::cout << "uBoundS " << uBoundX << " " << uBoundY << " " << uBoundZ << std::endl;
      std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ") ... cubeLen " << cubeLen/xscal << std::endl;

      std::cout << std::endl << "side " << side << " cubeSize " << size << " ite " << ite << std::endl << std::endl;

      gettimeofday(&begin, 0);
      initCubes();
      gettimeofday(&mid1, 0);
      initMergeTree();
      gettimeofday(&mid2, 0);

      // create the local merge trees for each cube
      gettimeofday(&mid3, 0);
      localStep();
      gettimeofday(&mid4, 0);

      std::cout << "-- localStep done" << std::endl;

      // merge local merge trees to create the global merge tree
      gettimeofday(&mid5, 0);
      globalStep();
      gettimeofday(&end, 0);

      std::cout << "-- globalStep done" << std::endl;

      checkValidMergeTree();
      getSizeOfMergeTree();
      writeMergeTreeToFile(filename);

      // sort particles again by their original particle id
      thrust::sort_by_key(idOriginal.begin(), idOriginal.end(), thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin(), leafVX.begin(), leafVY.begin(), leafVZ.begin(), leafM.begin(), leafI.begin(), leafParent.begin())));

      idOriginal.clear(); cubeId.clear();
      cubeMapping.clear(); particleSizeOfCubes.clear(); particleStartOfCubes.clear();
      edgesSrc.clear();  edgesDes.clear(); edgesWeight.clear(); edgeSizeOfCubes.clear(); edgeStartOfCubes.clear();
      tmpIntArray.clear(); tmpNxt.clear(); tmpFree.clear();

      std::cout << std::endl;
      timersub(&mid1, &begin, &diff1);
      float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
      std::cout << "Time elapsed: " << seconds1 << " s for initCubes"<< std::endl << std::flush;
      timersub(&mid2, &mid1, &diff2);
      float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
      std::cout << "Time elapsed: " << seconds2 << " s for initMergeTree"<< std::endl << std::flush;
      timersub(&mid4, &mid3, &diff3);
      float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
      std::cout << "Time elapsed: " << seconds3 << " s for localStep"<< std::endl << std::flush;
      timersub(&end, &mid5, &diff4);
      float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
      std::cout << "Time elapsed: " << seconds4 << " s for globalStep"<< std::endl << std::flush;
      totalTime = seconds1 + seconds2 + seconds3 + seconds4;
      std::cout << "Total time elapsed: " << totalTime << " s for constructing the global merge tree" << std::endl << std::endl;
    }
  }

  void operator()(float linkLength, int  particleSize)
  {
    clear();

    // Unnormalize linkLength so that it will work with box size distances
    linkLength   = linkLength*xscal;
    particleSize = particleSize;

    // if no valid particles, return
    if(numOfParticles==0) return;

    struct timeval begin, end, diff;

    gettimeofday(&begin, 0);
    findHalos(linkLength, particleSize); // find halos
    gettimeofday(&end, 0);

    timersub(&end, &begin, &diff);
    float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
    totalTime +=seconds;

    std::cout << "Total time elapsed: " << seconds << " s for finding halos at linking length " << linkLength/xscal << " and has particle size >= " << particleSize << std::endl << std::endl;

    getHaloDetails();   // get the unique halo ids & set numOfHalos
    getHaloParticles(); // get the halo particles & set numOfHaloParticles
    setColors();        // set colors to halos
    writeHaloResults(); // write halo results

    std::cout << "Number of Particles   : " << numOfParticles << std::endl;
    std::cout << "Number of Halos found : " << numOfHalos << std::endl;
    std::cout << "Merge tree size : " << mergetreeSize << std::endl;
    std::cout << "Min_ll  : " << min_ll/xscal  << std::endl;
    std::cout << "Max_ll  : " << max_ll/xscal << std::endl << std::endl;
    std::cout << "-----------------------------" << std::endl << std::endl;
  }



  //------- supporting functions - for merge tree

  // check whether the merge tree is valid or not
  void checkValidMergeTree()
  {
    tmpIntArray.resize(numOfParticles);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
          checkValid(thrust::raw_pointer_cast(&*tmpIntArray.begin()),
                     thrust::raw_pointer_cast(&*leafParent.begin()),
                     thrust::raw_pointer_cast(&*nodeParent.begin()),
                     thrust::raw_pointer_cast(&*nodeValue.begin()),
                     min_ll));

    int result = thrust::reduce(tmpIntArray.begin(), tmpIntArray.end());

    if(result!=0) std::cout << "-- ERROR: invalid merge tree " << std::endl;
    else std::cout << "-- valid merge tree " << std::endl;
  }

  // check whether a leafs parent nodes are valid
  struct checkValid : public thrust::unary_function<int, void>
  {
    double min_ll;
    int *tmpIntArray;

    int *leafParent, *nodeParent;
    float *nodeValue;

    __host__ __device__
    checkValid(int *tmpIntArray, int *leafParent, int *nodeParent, float *nodeValue, double min_ll) :
      tmpIntArray(tmpIntArray), leafParent(leafParent),
      nodeParent(nodeParent), nodeValue(nodeValue), min_ll(min_ll) {}

    __host__ __device__
    bool operator()(int i)
    {
      int tmp = leafParent[i];

      bool valid = (nodeValue[tmp]==min_ll);
      if(nodeParent[tmp]!=-1)
        valid &= (nodeValue[nodeParent[tmp]]>min_ll);

      tmpIntArray[i] = (valid) ? 0 : 1;
    }
  };

  // get the size of the merge tree
  void getSizeOfMergeTree()
  {
    tmpIntArray.resize(2*cubes);
    thrust::fill(tmpIntArray.begin(), tmpIntArray.end(), 0);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
          checkUsed(thrust::raw_pointer_cast(&*tmpIntArray.begin()),
                    thrust::raw_pointer_cast(&*leafParent.begin()),
                    thrust::raw_pointer_cast(&*nodeParent.begin())));

    mergetreeSize = thrust::reduce(tmpIntArray.begin(), tmpIntArray.end());
  }

  // check whether a leafs parent nodes are valid
  struct checkUsed : public thrust::unary_function<int, void>
  {
    int *tmpIntArray;

    int *leafParent, *nodeParent;

    __host__ __device__
    checkUsed(int *tmpIntArray, int *leafParent, int *nodeParent) :
      tmpIntArray(tmpIntArray), leafParent(leafParent), nodeParent(nodeParent) {}

    __host__ __device__
    bool operator()(int i)
    {
      int tmp = leafParent[i];

      tmpIntArray[tmp] = 1;
      while(nodeParent[tmp]!=-1)
      {
        tmp = nodeParent[tmp];
        tmpIntArray[tmp] = 1;
      }
    }
  };

  //write merge tree to a file
  void writeMergeTreeToFile(std::string filename)
  {
//    // write particle details - .particle file (pId x y z)
//    {
//      std::ofstream *outStream1 = new std::ofstream();
//      outStream1->open((filename+".particle").c_str()); //std::ios::out|std::ios::binary
//      (*outStream1) << "#pId x y z \n";
//
//      float fBlock[4];
//      for(int i=0; i<numOfParticles; i++)
//      {
//        fBlock[0] = i;   fBlock[1] = leafX[i];   fBlock[2] = leafY[i];   fBlock[3] = leafZ[i];
//
//        (*outStream1) << i << " ";
//        (*outStream1) << leafX[i] << " ";
//        (*outStream1) << leafY[i] << " ";
//        (*outStream1) << leafZ[i] << "\n";
//
//        //outStream1->write(reinterpret_cast<const char*>(fBlock), 4 * sizeof(float));
//      }
//      outStream1->close();
//    }
//
//    // count the valid nodes in the tree
//    tmpIntArray.resize(2*cubes);
//    thrust::for_each(CountingIterator(0), CountingIterator(0)+2*cubes,
//        checkIfNode(thrust::raw_pointer_cast(&*nodeI.begin()),
//                    thrust::raw_pointer_cast(&*tmpIntArray.begin())));
//    thrust::exclusive_scan(tmpIntArray.begin(), tmpIntArray.end(), tmpIntArray.begin(), 0);
//
//    // write feature details - .feature file (fId birth death parent offset size volume)
//    {
//      std::ofstream *outStream2 = new std::ofstream();
//      outStream2->open((filename+".feature").c_str());
//      (*outStream2) << "#fId birth death parent offset size volume \n";
//
//      int offset = 0;
//      for(int i=0; i<2*cubes; i++)
//        if(nodeValue[i]==min_ll)
//      {
//        if(nodeI[i]!=-1)
//        {
//          int size = 0;
//          if(nodeValue[i]==min_ll)
//          {
//            int child = nodeChildS[i];
//            while(child!=-1)
//            { child = leafSibling[child];   size++; }
//          }
//
//          (*outStream2) << tmpIntArray[i] << " ";
//          (*outStream2) << nodeValue[i]/xscal << " ";
//          (*outStream2) << ((nodeParent[i]!=-1) ? nodeValue[nodeParent[i]]/xscal : max_ll/xscal) << " ";
//          (*outStream2) << ((nodeParent[i]!=-1) ? tmpIntArray[nodeParent[i]] : -1) << " ";
//          (*outStream2) << offset << " " << size << " ";
//          (*outStream2) << nodeCount[i] << "\n";
//
//          offset += size;
//        }
//      }
//      outStream2->close();
//    }
//
//    // write segment details - .segmentation file (particlesIds)
//    {
//      std::ofstream *outStream3 = new std::ofstream();
//      outStream3->open((filename+".segmentation").c_str());
//      (*outStream3) << "#particlesIds \n";
//
//      for(int i=0; i<2*cubes; i++)
//      {
//        if(nodeI[i]!=-1)
//        {
//          if(nodeValue[i]==min_ll)
//          {
//            int child = nodeChildS[i];
//            while(child!=-1)
//            {
//              (*outStream3) << child << " ";
//              child = leafSibling[child];
//            }
//          }
//        }
//      }
//      outStream3->close();
//    }
  }



  //------- supporting functions - for halos

  // find halo ids
  void findHalos(float linkLength, int particleSize)
  {
    haloIndex.resize(numOfParticles);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
        setHaloId(thrust::raw_pointer_cast(&*leafParent.begin()),
                  thrust::raw_pointer_cast(&*leafParentS.begin()),
                  thrust::raw_pointer_cast(&*nodeValue.begin()),
                  thrust::raw_pointer_cast(&*nodeCount.begin()),
                  thrust::raw_pointer_cast(&*nodeI.begin()),
                  thrust::raw_pointer_cast(&*nodeParent.begin()),
                  thrust::raw_pointer_cast(&*haloIndex.begin()),
                  linkLength, particleSize));
  }

  // for a given node set its halo id, for pa 0 6 4 rticles in filtered halos set id to -1
  struct setHaloId : public thrust::unary_function<int, void>
  {
    int *leafParent, *leafParentS;

    float *nodeValue;
    int   *nodeCount, *nodeI, *nodeParent;
    int   *haloIndex;

    int    particleSize;
    float  linkLength;

    __host__ __device__
    setHaloId(int *leafParent, int *leafParentS,
      float *nodeValue, int *nodeCount, int *nodeI, int *nodeParent,
      int *haloIndex, float linkLength, int particleSize) :
      leafParent(leafParent), leafParentS(leafParentS),
      nodeValue(nodeValue), nodeCount(nodeCount), nodeI(nodeI), nodeParent(nodeParent),
      haloIndex(haloIndex), linkLength(linkLength), particleSize(particleSize) {}

    __host__ __device__
    void operator()(int i)
    {
      int n = leafParent[i];

      if(leafParentS[i]!=-1 && nodeValue[leafParentS[i]]<=linkLength)
        n = leafParentS[i];

      while(nodeParent[n]!=-1 && nodeValue[nodeParent[n]]<=linkLength)
        n = nodeParent[n];

      leafParentS[i] = n;

      haloIndex[i] = (nodeCount[n] >= particleSize) ? nodeI[n] : -1;
    }
  };

  // get the unique halo indexes & number of halos
  void getHaloDetails()
  {
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;

    // find unique halo ids & one particle id which belongs to that halo
    haloIndexUnique.resize(numOfParticles);
    thrust::copy(haloIndex.begin(), haloIndex.end(), haloIndexUnique.begin());
    tmpIntArray.resize(numOfParticles);
    thrust::sequence(tmpIntArray.begin(), tmpIntArray.end());

    thrust::stable_sort_by_key(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfParticles, tmpIntArray.begin(),  thrust::greater<int>());
    new_end = thrust::unique_by_key(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfParticles, tmpIntArray.begin());

    numOfHalos = thrust::get<0>(new_end) - haloIndexUnique.begin();
    if(haloIndexUnique[numOfHalos-1]==-1) numOfHalos--;

    thrust::reverse(tmpIntArray.begin(), tmpIntArray.begin()+numOfHalos);

    // get the halo stats
    haloCount.resize(numOfHalos);
    haloX.resize(numOfHalos);
    haloY.resize(numOfHalos);
    haloZ.resize(numOfHalos);
    haloVX.resize(numOfHalos);
    haloVY.resize(numOfHalos);
    haloVZ.resize(numOfHalos);

    thrust:: for_each(CountingIterator(0), CountingIterator(0)+numOfHalos,
        setHaloStats(thrust::raw_pointer_cast(&*leafParentS.begin()),
                     thrust::raw_pointer_cast(&*nodeCount.begin()),
                     thrust::raw_pointer_cast(&*nodeX.begin()),
                     thrust::raw_pointer_cast(&*nodeY.begin()),
                     thrust::raw_pointer_cast(&*nodeZ.begin()),
                     thrust::raw_pointer_cast(&*nodeVX.begin()),
                     thrust::raw_pointer_cast(&*nodeVY.begin()),
                     thrust::raw_pointer_cast(&*nodeVZ.begin()),
                     thrust::raw_pointer_cast(&*tmpIntArray.begin()),
                     thrust::raw_pointer_cast(&*haloCount.begin()),
                     thrust::raw_pointer_cast(&*haloX.begin()),
                     thrust::raw_pointer_cast(&*haloY.begin()),
                     thrust::raw_pointer_cast(&*haloZ.begin()),
                     thrust::raw_pointer_cast(&*haloVX.begin()),
                     thrust::raw_pointer_cast(&*haloVY.begin()),
                     thrust::raw_pointer_cast(&*haloVZ.begin())));
  }

  // for each halo, get its stats
  struct setHaloStats : public thrust::unary_function<int, void>
  {
    int *nodeCount, *leafParentS;
    float *nodeX, *nodeY, *nodeZ;
    float *nodeVX, *nodeVY, *nodeVZ;

    int *particleId;
    int *haloCount;
    float *haloX, *haloY, *haloZ;
    float *haloVX, *haloVY, *haloVZ;

    __host__ __device__
    setHaloStats(int *leafParentS, int *nodeCount,
      float *nodeX, float *nodeY, float *nodeZ,
      float *nodeVX, float *nodeVY, float *nodeVZ,
      int *particleId, int *haloCount,
      float *haloX, float *haloY, float *haloZ,
      float *haloVX, float *haloVY, float *haloVZ) :
      leafParentS(leafParentS), nodeCount(nodeCount),
      nodeX(nodeX), nodeY(nodeY), nodeZ(nodeZ),
      nodeVX(nodeVX), nodeVY(nodeVY), nodeVZ(nodeVZ),
      particleId(particleId), haloCount(haloCount),
      haloX(haloX), haloY(haloY), haloZ(haloZ),
      haloVX(haloVX), haloVY(haloVY), haloVZ(haloVZ) {}

    __host__ __device__
    void operator()(int i)
    {
      int n = leafParentS[particleId[i]];

      haloCount[i] = nodeCount[n];
      haloX[i] = (float)(nodeX[n]/nodeCount[n]);  haloVX[i] = (float)(nodeVX[n]/nodeCount[n]);
      haloY[i] = (float)(nodeY[n]/nodeCount[n]);  haloVY[i] = (float)(nodeVY[n]/nodeCount[n]);
      haloZ[i] = (float)(nodeZ[n]/nodeCount[n]);  haloVZ[i] = (float)(nodeVZ[n]/nodeCount[n]);
    }
  };

  // get particles of valid halos & get number of halo particles
  void getHaloParticles()
  {
    thrust::device_vector<int>::iterator new_end;

    tmpIntArray.resize(numOfParticles);
    thrust::sequence(tmpIntArray.begin(), tmpIntArray.begin()+numOfParticles);

    new_end = thrust::remove_if(tmpIntArray.begin(), tmpIntArray.begin()+numOfParticles,
        invalidHalo(thrust::raw_pointer_cast(&*haloIndex.begin())));

    numOfHaloParticles = new_end - tmpIntArray.begin();

    haloIndex_f.resize(numOfHaloParticles);
    inputX_f.resize(numOfHaloParticles);
    inputY_f.resize(numOfHaloParticles);
    inputZ_f.resize(numOfHaloParticles);

    thrust::gather(tmpIntArray.begin(), tmpIntArray.begin()+numOfHaloParticles, haloIndex.begin(), haloIndex_f.begin());

    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfHaloParticles,
      getHaloParticlePositions(thrust::raw_pointer_cast(&*leafX.begin()),
                               thrust::raw_pointer_cast(&*leafY.begin()),
                               thrust::raw_pointer_cast(&*leafZ.begin()),
                               thrust::raw_pointer_cast(&*tmpIntArray.begin()),
                               thrust::raw_pointer_cast(&*inputX_f.begin()),
                               thrust::raw_pointer_cast(&*inputY_f.begin()),
                               thrust::raw_pointer_cast(&*inputZ_f.begin())));
  }

  // given a haloIndex of a particle, check whether this particle DOES NOT belong to a halo
  struct invalidHalo : public thrust::unary_function<int, bool>
  {
    int  *haloIndex;

    __host__ __device__
    invalidHalo(int *haloIndex) : haloIndex(haloIndex) {}

    __host__ __device__
    bool operator()(int i)
    {
      return (haloIndex[i]==-1);
    }
  };

  // for each particle in a halo, get its positions
  struct getHaloParticlePositions : public thrust::unary_function<int, void>
  {
    int *particleId;

    float *leafX, *leafY, *leafZ;
    float *inputX_f, *inputY_f, *inputZ_f;

    __host__ __device__
    getHaloParticlePositions(float *leafX, float *leafY, float *leafZ, int *particleId,
      float *inputX_f, float *inputY_f, float *inputZ_f) :
      leafX(leafX), leafY(leafY), leafZ(leafZ), particleId(particleId),
      inputX_f(inputX_f), inputY_f(inputY_f), inputZ_f(inputZ_f) {}

    __host__ __device__
    void operator()(int i)
    {
      int n = particleId[i];

      inputX_f[i] = leafX[n];
      inputY_f[i] = leafY[n];
      inputZ_f[i] = leafZ[n];
    }
  };



  //------- initialization step

  // divide space into cubes
  void initCubes()
  {
    // compute which cube each particle is in
    cubeId.resize(numOfParticles);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
        setCubeIdOfParticle(thrust::raw_pointer_cast(&*leafX.begin()),
                            thrust::raw_pointer_cast(&*leafY.begin()),
                            thrust::raw_pointer_cast(&*leafZ.begin()),
                            thrust::raw_pointer_cast(&*cubeId.begin()),
                            cubeLen, lBoundX, lBoundY, lBoundZ, cubesInX, cubesInY, cubesInZ));

    // sort particles by cube
    idOriginal.resize(numOfParticles);
    thrust::sequence(idOriginal.begin(), idOriginal.end(), 0);
    thrust::sort_by_key(cubeId.begin(), cubeId.end(), thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin(), leafVX.begin(), leafVY.begin(), leafVZ.begin(), leafM.begin(), leafI.begin(), idOriginal.begin())));

    // get the size,start & cube mapping details for only non empty cubes
    int num = (numOfParticles<numOfCubes) ? numOfParticles : numOfCubes;
    cubeMapping.resize(num);
    particleSizeOfCubes.resize(num);
    thrust::pair<thrust::device_vector<unsigned int>::iterator, thrust::device_vector<int>::iterator> new_end;
    new_end = thrust::reduce_by_key(cubeId.begin(), cubeId.end(), ConstantIterator(1), cubeMapping.begin(), particleSizeOfCubes.begin());

    unsigned int cubesNonEmpty = thrust::get<0>(new_end) - cubeMapping.begin();
    unsigned int cubesEmpty    = (numOfCubes - cubesNonEmpty);

    std::cout << cubesEmpty << " of " << numOfCubes << " cubes are empty. (" << (((double)cubesEmpty*100)/(double)numOfCubes) << "%) ... non empty cubes " << cubesNonEmpty << std::endl;

    cubes = cubesNonEmpty; // get the cubes which should be considered

    cubeMapping.resize(cubes);
    particleSizeOfCubes.resize(cubes);
    particleStartOfCubes.resize(cubes);
    thrust::exclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes, particleStartOfCubes.begin());

    #ifdef TEST
      outputCubeDetails("cube details"); // output cube details
    #endif
  }

  // for a given particle, set its cube id
  struct setCubeIdOfParticle : public thrust::unary_function<int, void>
  {
    float  cubeLen;
    float  lBoundX, lBoundY, lBoundZ;
    int    cubesInX, cubesInY, cubesInZ;

    unsigned int *cubeId;
    float *leafX, *leafY, *leafZ;

    __host__ __device__
    setCubeIdOfParticle(float *leafX, float *leafY, float *leafZ,
      unsigned int *cubeId, float cubeLen,
      float lBoundX, float lBoundY, float lBoundZ,
      int cubesInX, int cubesInY, int cubesInZ) :
      leafX(leafX), leafY(leafY), leafZ(leafZ),
      cubeId(cubeId), cubeLen(cubeLen),
      lBoundX(lBoundX), lBoundY(lBoundY), lBoundZ(lBoundZ),
      cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ){}

    __host__ __device__
    void operator()(int i)
    {
      // get x,y,z coordinates for the cube
      int x = (leafX[i]-lBoundX)/cubeLen; if(x>=cubesInX) x = cubesInX-1;
      int y = (leafY[i]-lBoundY)/cubeLen; if(y>=cubesInY) y = cubesInY-1;
      int z = (leafZ[i]-lBoundZ)/cubeLen; if(z>=cubesInZ) z = cubesInZ-1;

      cubeId[i] = x + y*cubesInX + z*(cubesInX*cubesInY); // get cube id
    }
  };

  // for a given particle, set its cube size
  struct get_cube_counts : public thrust::unary_function<int, void>
  {
    int  numOfParticles;

    int* cubeId;
    int* tmpIntArray;
    int* particleSizeOfCubes;

    get_cube_counts(int* cubeId, int* tmpIntArray, int* particleSizeOfCubes, int numOfParticles) :
      cubeId(cubeId), tmpIntArray(tmpIntArray), particleSizeOfCubes(particleSizeOfCubes), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator() (int i)
    {
      if((i == numOfParticles-1) || (cubeId[i] != cubeId[i+1]))
        particleSizeOfCubes[cubeId[i]] = tmpIntArray[i];
    }
  };

  // initialize arrays for storing the merge tree
  void initMergeTree()
  {
    // set leaf details
    leafParent.resize(numOfParticles);
    leafParentS.resize(numOfParticles);

    thrust::fill(leafParentS.begin(), leafParentS.end(), -1);

    // set node details
    nodeX.resize(2*cubes);
    nodeY.resize(2*cubes);
    nodeZ.resize(2*cubes);
    nodeVX.resize(2*cubes);
    nodeVY.resize(2*cubes);
    nodeVZ.resize(2*cubes);
    nodeM.resize(2*cubes);
    nodeI.resize(2*cubes);
    nodeValue.resize(2*cubes);
    nodeCount.resize(2*cubes);
    nodeParent.resize(2*cubes);

    thrust::fill(nodeParent.begin(), nodeParent.end(), -1);
  }



  //------- local step

  // locally, create the local merge trees for each cube
  void localStep()
  {
    struct timeval begin, mid1, mid2, mid3, end, diff1, diff2, diff3, diff4;

    gettimeofday(&begin, 0);
    initArrays(); // init arrays needed for storing edges
    gettimeofday(&mid1, 0);
    getEdgesPerCube(); // for each cube, get the set of edges
    gettimeofday(&mid2, 0);
    // set vectors necessary for merge tree construction
    tmpNxt.resize(cubes);
    tmpFree.resize(2*cubes);
    thrust::sequence(tmpFree.begin(), tmpFree.end(), 1);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        initNxt(thrust::raw_pointer_cast(&*tmpNxt.begin()),
                thrust::raw_pointer_cast(&*tmpFree.begin())));
    gettimeofday(&mid3, 0);
    // set local merge trees for each cube
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        createSubMergeTree(thrust::raw_pointer_cast(&*leafParent.begin()),
                           thrust::raw_pointer_cast(&*leafI.begin()),
                           thrust::raw_pointer_cast(&*leafX.begin()),
                           thrust::raw_pointer_cast(&*leafY.begin()),
                           thrust::raw_pointer_cast(&*leafZ.begin()),
                           thrust::raw_pointer_cast(&*leafVX.begin()),
                           thrust::raw_pointer_cast(&*leafVY.begin()),
                           thrust::raw_pointer_cast(&*leafVZ.begin()),
                           thrust::raw_pointer_cast(&*nodeI.begin()),
                           thrust::raw_pointer_cast(&*nodeCount.begin()),
                           thrust::raw_pointer_cast(&*nodeValue.begin()),
                           thrust::raw_pointer_cast(&*nodeX.begin()),
                           thrust::raw_pointer_cast(&*nodeY.begin()),
                           thrust::raw_pointer_cast(&*nodeZ.begin()),
                           thrust::raw_pointer_cast(&*nodeVX.begin()),
                           thrust::raw_pointer_cast(&*nodeVY.begin()),
                           thrust::raw_pointer_cast(&*nodeVZ.begin()),
                           thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
                           thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
                           thrust::raw_pointer_cast(&*tmpNxt.begin()),
                           thrust::raw_pointer_cast(&*tmpFree.begin()),
                           min_ll));
    gettimeofday(&end, 0);

    binStart.clear();

    std::cout << std::endl;
    std::cout << "'localStep' Time division:" << std::endl << std::flush;
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed0: " << seconds1 << " s for initArrays" << std::endl << std::flush;
    timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed1: " << seconds2 << " s for getEdgesPerCube" << std::endl << std::flush;
    timersub(&mid3, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed2: " << seconds3 << " s for initNxt" << std::endl << std::flush;
    timersub(&end, &mid3, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed3: " << seconds4 << " s for createSubMergeTree" << std::endl << std::flush;
  }

  // for each cube, init arrays needed for storing edges
  void initArrays()
  {
    // set bins
    binSize = cubesInX;
    bins = std::ceil((double)numOfCubes/binSize);

    std::cout << std::endl << "bins " << std::ceil((double)numOfCubes/n) << " binSize " << binSize << std::endl;

    binStart.resize(std::ceil((double)numOfCubes/binSize));
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        setBins(thrust::raw_pointer_cast(&*cubeMapping.begin()),
                thrust::raw_pointer_cast(&*binStart.begin()),
                binSize));

    std::cout << std::endl << "bins " << std::ceil((double)numOfCubes/n) << " binSize " << binSize << std::endl;

//--------------------------------------------
//    // get size of edges
//    numOfEdges = ite*cubes;
//
//    std::cout << std::endl << "numOfEdges before " << numOfEdges << std::endl;
//
//    // init edge arrays
//    edgesSrc.resize(numOfEdges);
//    edgesDes.resize(numOfEdges);
//    edgesWeight.resize(numOfEdges);
//
//    // for each cube, set the space required for storing edges
//    edgeSizeOfCubes.resize(cubes);
//    edgeStartOfCubes.resize(cubes);
//    thrust::fill(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end(), ite);
//    thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes, edgeStartOfCubes.begin());
//--------------------------------------------
    edgeSizeOfCubes.resize(cubes);
    edgeStartOfCubes.resize(cubes);

    tmpIntArray.resize(cubes);
    // for each cube, get neighbor details
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        getNeighborDetails(thrust::raw_pointer_cast(&*cubeMapping.begin()),
                           thrust::raw_pointer_cast(&*binStart.begin()),
                           thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
                           thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                           thrust::raw_pointer_cast(&*tmpIntArray.begin()),
                           ite, side, cubesInX, cubesInY, cubesInZ, cubes, binSize, bins));
    //set chunks for load balance
    setChunks();
    tmpIntArray.clear();

    // for each cube, set the space required for storing edges
    thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes, edgeStartOfCubes.begin());
    numOfEdges = edgeStartOfCubes[cubes-1] + edgeSizeOfCubes[cubes-1]; // size of edges array

    // init edge arrays
    edgesSrc.resize(numOfEdges);
    edgesDes.resize(numOfEdges);
    edgesWeight.resize(numOfEdges);

    std::cout << std::endl << "numOfEdges before " << numOfEdges << std::endl;
//--------------------------------------------
  }

  // for each bin, set the cubeStart in cubeMapping
  struct setBins : public thrust::unary_function<int, void>
  {
    int  binSize;
    int *binStart;
    unsigned int *cubeMapping;

    __host__ __device__
    setBins(unsigned int *cubeMapping, int *binStart, int binSize) :
      cubeMapping(cubeMapping), binStart(binStart), binSize(binSize) {}

    __host__ __device__
    void operator()(int i)
    {
      //std::cout << i << std::endl;
      unsigned int n = cubeMapping[i]/binSize;
      //std::cout << i << " " << n << " " << cubeMapping[i] << std::endl;

      if(i==0)
      {
        binStart[n] = i;
        return;
      }

      unsigned int m = cubeMapping[i-1]/binSize;
      if(m!=n)
      {
        for(unsigned int j=m+1; j<=n; j++)
          binStart[j] = i;
      }
    }
  };

  //for each cube, sum the number of particles in neighbor cubes & get the sum of non empty neighbor cubes
  struct getNeighborDetails : public thrust::unary_function<int, void>
  {
    unsigned int *cubeMapping;
    int *particleSizeOfCubes;
    int *tmpIntArray, *edgeSizeOfCubes;

    int  binSize, bins;
    int *binStart;

    int  ite, side;
    int  cubesInX, cubesInY, cubesInZ, cubes;

    __host__ __device__
    getNeighborDetails(unsigned int *cubeMapping, int *binStart, int *particleSizeOfCubes,
        int *edgeSizeOfCubes, int *tmpIntArray, int ite, int side,
        int cubesInX, int cubesInY, int cubesInZ, int cubes, int binSize, int bins) :
        cubeMapping(cubeMapping), binStart(binStart), particleSizeOfCubes(particleSizeOfCubes),
        edgeSizeOfCubes(edgeSizeOfCubes), tmpIntArray(tmpIntArray), ite(ite), side(side),
        cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), cubes(cubes), binSize(binSize), bins(bins) {}

    __host__ __device__
    void operator()(int i)
    {
      int i_mapped = cubeMapping[i];

      // get x,y,z coordinates for the cube
      const int x =  i_mapped % cubesInX;
      const int y = (i_mapped / cubesInX) % cubesInY;
      const int z =  i_mapped / (cubesInX*cubesInY);

      int len = (side-1)/2;

      int sumNonEmptyCubes = 0;
      int sumParticles = 0;

      for(int num=0; num<ite; num++)
      {
        const int x1 =  num % side;
        const int y1 = (num / side) % side;
        const int z1 =  num / (side*side);

        // get x,y,z coordinates for the current cube
        int currentX = x - len + x1;
        int currentY = y - len + y1;
        int currentZ = z - len + z1;

        int cube_mapped = -1, cube = -1;
        if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
        {
          cube_mapped = currentX  + currentY*cubesInX + currentZ*(cubesInY*cubesInX);

          int bin = cube_mapped/binSize;

          int from = binStart[bin];
          int to   = (bin+1 < bins) ? binStart[bin+1]-1 : cubes;

//            int from = (i_mapped>cube_mapped) ? 0 : l+1;
//            int to   = (i_mapped>cube_mapped) ? l-1 : cubes-1;

          while(to >= from)
          {
            int mid = from + ((to - from) / 2);
            if (cubeMapping[mid] < cube_mapped)
              from = mid + 1;
            else if (cubeMapping[mid] > cube_mapped)
              to = mid - 1;
            else
            { cube = mid; break; }
          }
        }

        if(cube_mapped==-1 || cube==-1) continue;

        sumNonEmptyCubes++;  //sum the non empty neighbor cubes
        sumParticles += particleSizeOfCubes[cube]; // sum the number of particles in neighbor cubes
      }

      tmpIntArray[i] = (sumParticles*particleSizeOfCubes[i]); // store number of comparisons for this cube
      edgeSizeOfCubes[i] = sumNonEmptyCubes; // store sum of non empty neighbor cubes for this cube
    }
  };

//------------ TODO
  // group the set of cubes to chunks of same computation sizes (for load balance)
  void setChunks()
  {
    float maxSize = *(thrust::max_element(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes));
//    float maxSize = *(thrust::max_element(tmpIntArray.begin(), tmpIntArray.begin()+cubes));

    startOfChunks.resize(cubes);
    sizeOfChunks.resize(cubes);

    thrust::fill(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
    thrust::inclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes, sizeOfChunks.begin());
//    thrust::inclusive_scan(tmpIntArray.begin(), tmpIntArray.begin()+cubes, sizeOfChunks.begin());

//    std::cout << "particleSizeOfCubes  "; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//    std::cout << "inclusive_scan   "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;

    thrust::copy_if(CountingIterator(0), CountingIterator(0)+cubes, startOfChunks.begin(),
        isStartOfChunks(thrust::raw_pointer_cast(&*sizeOfChunks.begin()), maxSize));
    chunks = cubes - thrust::count(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
        setSizeOfChunks(thrust::raw_pointer_cast(&*startOfChunks.begin()),
                        thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
                        chunks, cubes));

    std::cout << "maxSize " << maxSize << " chunks " << chunks << std::endl;
//    std::cout << "startOfChunks  "; thrust::copy(startOfChunks.begin(), startOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//    std::cout << "sizeOfChunks   "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//    std::cout << "amountOfChunks   ";
//    for(int i=0; i<chunks; i++)
//    {
//      int count = 0;
//      for(int j=startOfChunks[i]; j<startOfChunks[i]+sizeOfChunks[i]; j++)
//        count += particleSizeOfCubes[j];
//      std::cout << count << " ";
//    }
//    std::cout << std::endl << std::endl;
  }

  // check whether this cube is the start of a chunk
  struct isStartOfChunks : public thrust::unary_function<int, void>
  {
    int max;
    int *sizeOfChunks;

    __host__ __device__
    isStartOfChunks(int *sizeOfChunks, int max) :
      sizeOfChunks(sizeOfChunks), max(max) {}

    __host__ __device__
    bool operator()(int i)
    {
      if(i==0) return true;

      int a = sizeOfChunks[i] / max;  int b = sizeOfChunks[i-1] / max;
      int d = sizeOfChunks[i] % max;  int e = sizeOfChunks[i-1] % max;

      if((a!=b && d==0 && e==0) || (a!=b && d!=0) || (a==b && e==0)) return true;

      return false;
    }
  };

  // set the size of cube
  struct setSizeOfChunks : public thrust::unary_function<int, void>
  {
    int chunks, cubes;
    int *startOfChunks, *sizeOfChunks;

    __host__ __device__
    setSizeOfChunks(int *startOfChunks, int *sizeOfChunks, int chunks, int cubes) :
      startOfChunks(startOfChunks), sizeOfChunks(sizeOfChunks), chunks(chunks), cubes(cubes) {}

    __host__ __device__
    void operator()(int i)
    {
      if(i==chunks-1) sizeOfChunks[i] = cubes - startOfChunks[i];
      else            sizeOfChunks[i] = startOfChunks[i+1] - startOfChunks[i];
    }
  };

  // for each cube, get the set of edges by running them in chunks of cubes
  void getEdgesPerCube()
  {
    thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
        getEdges(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
                 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
                 thrust::raw_pointer_cast(&*startOfChunks.begin()),
                 thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
                 thrust::raw_pointer_cast(&*cubeMapping.begin()),
                 thrust::raw_pointer_cast(&*binStart.begin()),
                 thrust::raw_pointer_cast(&*leafX.begin()),
                 thrust::raw_pointer_cast(&*leafY.begin()),
                 thrust::raw_pointer_cast(&*leafZ.begin()),
                 max_ll, min_ll, ite, cubesInX, cubesInY, cubesInZ, cubes, side, binSize, bins,
                 thrust::raw_pointer_cast(&*edgesSrc.begin()),
                 thrust::raw_pointer_cast(&*edgesDes.begin()),
                 thrust::raw_pointer_cast(&*edgesWeight.begin()),
                 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));

    numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes); // set the correct number of edges

    std::cout << "numOfEdges after " << numOfEdges << std::endl;
  }

//------------ TODO
  // for each cube, get the set of edges after comparing
  struct getEdges : public thrust::unary_function<int, void>
  {
    int    ite;
    float  max_ll, min_ll;
    float *leafX, *leafY, *leafZ;

    int   *startOfChunks, *sizeOfChunks;
    unsigned int *cubeMapping;
    int   *particleStartOfCubes, *particleSizeOfCubes;

    int    binSize, bins;
    int   *binStart;

    int    side;
    int    cubesInX, cubesInY, cubesInZ, cubes;

    int   *edgesSrc, *edgesDes;
    float *edgesWeight;
    int   *edgeStartOfCubes, *edgeSizeOfCubes;

    __host__ __device__
    getEdges(int *particleStartOfCubes, int *particleSizeOfCubes,
        int *startOfChunks, int *sizeOfChunks,
        unsigned int *cubeMapping, int *binStart,
        float *leafX, float *leafY, float *leafZ,
        float max_ll, float min_ll, int ite,
        int cubesInX, int cubesInY, int cubesInZ, int cubes, int side,
        int binSize, int bins,
        int *edgesSrc, int *edgesDes, float *edgesWeight,
        int *edgeSizeOfCubes, int *edgeStartOfCubes) :
        particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
        startOfChunks(startOfChunks), sizeOfChunks(sizeOfChunks),
        cubeMapping(cubeMapping), binStart(binStart),
        leafX(leafX), leafY(leafY), leafZ(leafZ),
        max_ll(max_ll), min_ll(min_ll), ite(ite),
        cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), cubes(cubes), side(side),
        binSize(binSize), bins(bins),
        edgesSrc(edgesSrc), edgesDes(edgesDes), edgesWeight(edgesWeight),
        edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

    __host__ __device__
    void operator()(int i)
    {
      for(int l=startOfChunks[i]; l<startOfChunks[i]+sizeOfChunks[i]; l++)
      {
        int i_mapped = cubeMapping[l];

        // get x,y,z coordinates for the cube
        const int x =  i_mapped % cubesInX;
        const int y = (i_mapped / cubesInX) % cubesInY;
        const int z =  i_mapped / (cubesInX*cubesInY);

        int len = (side-1)/2;

        int size = 0;
        int start = edgeStartOfCubes[l];

        for(int num=0; num<ite; num++)
        {
          const int x1 =  num % side;
          const int y1 = (num / side) % side;
          const int z1 =  num / (side*side);

          // get x,y,z coordinates for the current cube
          int currentX = x - len + x1;
          int currentY = y - len + y1;
          int currentZ = z - len + z1;

          int cube_mapped = -1, cube = -1;
          if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
          {
            cube_mapped = currentX  + currentY*cubesInX + currentZ*(cubesInY*cubesInX);

            int bin = cube_mapped/binSize;

            int from = binStart[bin];
            int to   = (bin+1 < bins) ? binStart[bin+1]-1 : cubes;

//            int from = (i_mapped>cube_mapped) ? 0 : l+1;
//            int to   = (i_mapped>cube_mapped) ? l-1 : cubes-1;

            while(to >= from)
            {
              int mid = from + ((to - from) / 2);
              if (cubeMapping[mid] < cube_mapped)
                from = mid + 1;
              else if (cubeMapping[mid] > cube_mapped)
                to = mid - 1;
              else
              { cube = mid; break; }
            }
          }

          if(cube_mapped==-1 || cube==-1) continue;

          // for each particle in this cube
          int eSrc, eDes;
          float eWeight = max_ll+1;
          for(int j=particleStartOfCubes[l]; j<particleStartOfCubes[l]+particleSizeOfCubes[l]; j++)
          {
            float3 p_j = make_float3(leafX[j], leafY[j], leafZ[j]);

            // compare with particles in neighboring cube
            for(int k=particleStartOfCubes[cube]; k<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; k++)
            {
              float3 p_k = make_float3(leafX[k], leafY[k], leafZ[k]);

              double dist = (double)std::sqrt((p_j.x-p_k.x)*(p_j.x-p_k.x) + (p_j.y-p_k.y)*(p_j.y-p_k.y) + (p_j.z-p_k.z)*(p_j.z-p_k.z));
              if(dist <= max_ll && dist < eWeight)
              {
                eSrc = j; eDes = k; eWeight = dist;
                if(eWeight <= min_ll) goto loop;
              }
            }
          }

          // add edge
          loop:
          if(eWeight < max_ll + 1)
          {
            edgesSrc[start + size] = eSrc;
            edgesDes[start + size] = eDes;
            edgesWeight[start + size] = eWeight;
            size++;
          }
        }

        edgeSizeOfCubes[l] = size;
      }
    }
  };

  // finalize the init of tmpFree & tmpNxt arrays
  struct initNxt : public thrust::unary_function<int, void>
  {
    int *tmpFree, *tmpNxt;

    __host__ __device__
    initNxt(int *tmpNxt, int *tmpFree) : tmpNxt(tmpNxt), tmpFree(tmpFree) {}

    __host__ __device__
    void operator()(int i)
    {
      tmpNxt[i] = 2*i;
      tmpFree[2*i+1] = -1;
    }
  };

  // create the submerge tree for each cube
  struct createSubMergeTree : public thrust::unary_function<int, void>
  {
    int   *leafParent;
    int   *leafI;
    float *leafX, *leafY, *leafZ;
    float *leafVX, *leafVY, *leafVZ;

    int   *nodeI, *nodeCount;
    float *nodeValue;
    float *nodeX, *nodeY, *nodeZ;
    float *nodeVX, *nodeVY, *nodeVZ;

    int *particleSizeOfCubes, *particleStartOfCubes;
    int *tmpNxt, *tmpFree;

    float min_ll;

    __host__ __device__
    createSubMergeTree(int *leafParent, int *leafI,
      float *leafX, float *leafY, float *leafZ,
      float *leafVX, float *leafVY, float *leafVZ,
      int *nodeI, int *nodeCount, float *nodeValue,
      float *nodeX, float *nodeY, float *nodeZ,
      float *nodeVX, float *nodeVY, float *nodeVZ,
      int *particleSizeOfCubes, int *particleStartOfCubes,
      int *tmpNxt, int *tmpFree, float min_ll) :
      leafParent(leafParent), leafI(leafI),
      leafX(leafX), leafY(leafY), leafZ(leafZ),
      leafVX(leafVX), leafVY(leafVY), leafVZ(leafVZ),
      nodeI(nodeI), nodeCount(nodeCount), nodeValue(nodeValue),
      nodeX(nodeX), nodeY(nodeY), nodeZ(nodeZ),
      nodeVX(nodeVX), nodeVY(nodeVY), nodeVZ(nodeVZ),
      particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes),
      tmpNxt(tmpNxt), tmpFree(tmpFree), min_ll(min_ll) {}

    __host__ __device__
    void operator()(int i)
    {
      // get the next free node & set it as the parent
      int n = tmpNxt[i];
      int tmpVal = tmpFree[tmpNxt[i]];
      tmpFree[tmpNxt[i]] = -2;
      tmpNxt[i] = tmpVal;

      float x=0,  y=0,  z=0;
      float vx=0, vy=0, vz=0;

      int minHaloId = -1;
      for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
      {
        leafParent[j] = n; // n is the index in node lists

        minHaloId = (minHaloId==-1) ? leafI[j] : (minHaloId<leafI[j] ? minHaloId : leafI[j]);

        x+=leafX[j]; vx+=leafVX[j];
        y+=leafY[j]; vy+=leafVY[j];
        z+=leafZ[j]; vz+=leafVZ[j];
      }

      nodeI[n] = minHaloId;
      nodeValue[n] = min_ll;
      nodeCount[n] = particleSizeOfCubes[i];
      nodeX[n] = x;   nodeVX[n] = vx;
      nodeY[n] = y;   nodeVY[n] = vy;
      nodeZ[n] = z;   nodeVZ[n] = vz;
    }
  };



  //------- global step

  // merge local merge trees to create the global merge tree
  void globalStep()
  {
    int cubesOri = cubes;
    int sizeP = 2;
    int i = 0;

    // set new number of cubes
    int cubesOld = cubes;
    cubes = (int)std::ceil(((double)cubes/2));

    // iteratively combine the cubes two at a time
    while(cubes!=cubesOld && cubes>0)
    {
      struct timeval begin, mid1, mid2, mid3, end, diff, diff1, diff2, diff3, diff4;
      gettimeofday(&begin, 0);
      thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        combineFreeLists(thrust::raw_pointer_cast(&*tmpNxt.begin()),
                         thrust::raw_pointer_cast(&*tmpFree.begin()),
                         sizeP, cubesOri));
      gettimeofday(&mid1, 0);
      thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        combineMergeTrees(thrust::raw_pointer_cast(&*cubeMapping.begin()),
                          thrust::raw_pointer_cast(&*cubeId.begin()),
                          thrust::raw_pointer_cast(&*tmpNxt.begin()),
                          thrust::raw_pointer_cast(&*tmpFree.begin()),
                          thrust::raw_pointer_cast(&*leafParent.begin()),
                          thrust::raw_pointer_cast(&*nodeParent.begin()),
                          thrust::raw_pointer_cast(&*nodeI.begin()),
                          thrust::raw_pointer_cast(&*nodeValue.begin()),
                          thrust::raw_pointer_cast(&*nodeCount.begin()),
                          thrust::raw_pointer_cast(&*nodeX.begin()),
                          thrust::raw_pointer_cast(&*nodeY.begin()),
                          thrust::raw_pointer_cast(&*nodeZ.begin()),
                          thrust::raw_pointer_cast(&*nodeVX.begin()),
                          thrust::raw_pointer_cast(&*nodeVY.begin()),
                          thrust::raw_pointer_cast(&*nodeVZ.begin()),
                          thrust::raw_pointer_cast(&*edgesSrc.begin()),
                          thrust::raw_pointer_cast(&*edgesDes.begin()),
                          thrust::raw_pointer_cast(&*edgesWeight.begin()),
                          thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
                          thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                          min_ll, sizeP, cubesOri));
      gettimeofday(&mid2, 0);
      thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
          jumpLeafPointers(thrust::raw_pointer_cast(&*leafParent.begin()),
                           thrust::raw_pointer_cast(&*nodeParent.begin()),
                           thrust::raw_pointer_cast(&*nodeCount.begin()),
                           thrust::raw_pointer_cast(&*nodeValue.begin())));
      gettimeofday(&mid3, 0);
      thrust::for_each(CountingIterator(0), CountingIterator(0)+2*cubesOri,
          jumpNodePointers(thrust::raw_pointer_cast(&*nodeParent.begin()),
                           thrust::raw_pointer_cast(&*nodeCount.begin()),
                           thrust::raw_pointer_cast(&*nodeValue.begin())));
      gettimeofday(&end, 0);

//      std::cout << std::endl;
//      std::cout << "'globalStep' Time division:" << std::endl << std::flush;
//      timersub(&mid1, &begin, &diff1);
//      float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
//      std::cout << "Time elapsed0: " << seconds1 << " s for combineFreeLists" << std::endl << std::flush;
//      timersub(&mid2, &mid1, &diff2);
//      float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
//      std::cout << "Time elapsed1: " << seconds2 << " s for combineMergeTrees" << std::endl << std::flush;
//      timersub(&mid3, &mid2, &diff3);
//      float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
//      std::cout << "Time elapsed2: " << seconds3 << " s for jumpLeafPointers" << std::endl << std::flush;
//      timersub(&end, &mid3, &diff4);
//      float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
//      std::cout << "Time elapsed3: " << seconds4 << " s for jumpNodePointers" << std::endl << std::flush;

      timersub(&end, &begin, &diff);
      float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
      std::cout << "Iteration: " << i << " ... time elapsed " << seconds << "s nonEmptyCubes " << cubes << " numOfEdges " << numOfEdges << std::endl;

      numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubesOri);

      // set new number of cubes & sizeP
      i++;
      sizeP *= 2;
      cubesOld = cubes;
      cubes = (int)std::ceil(((double)cubes/2));
    }

    cubes = cubesOri;
  }

  // combine free nodes lists of each cube at this iteration
  struct combineFreeLists : public thrust::unary_function<int, void>
  {
    int  sizeP, numOfCubesOri;

    int *tmpNxt, *tmpFree;

    __host__ __device__
    combineFreeLists(int *tmpNxt, int *tmpFree, int sizeP, int numOfCubesOri) :
        tmpNxt(tmpNxt), tmpFree(tmpFree), sizeP(sizeP), numOfCubesOri(numOfCubesOri) {}

    __host__ __device__
    void operator()(int i)
    {
      int cubeStart = sizeP*i;
      int cubeEnd   = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

      int k;
      for(k=cubeStart; k<cubeEnd; k+=sizeP/2)
      { if(tmpNxt[k]!=-1) { tmpNxt[cubeStart] = tmpNxt[k];  break; } }

      int nxt;
      while(k<cubeEnd)
      {
        nxt = tmpNxt[k];

        while(nxt!=-1 && tmpFree[nxt]!=-1)
          nxt = tmpFree[nxt];

        k += sizeP/2;
        if(k<cubeEnd) tmpFree[nxt] = tmpNxt[k];
      }
    }
  };

  // combine two local merge trees
  struct combineMergeTrees : public thrust::unary_function<int, void>
  {
    float  min_ll;
    int    sizeP, numOfCubesOri;

    unsigned int *cubeId;
    unsigned int *cubeMapping;
    int   *tmpNxt, *tmpFree;

    int   *edgesSrc, *edgesDes;
    float *edgesWeight;
    int   *edgeStartOfCubes, *edgeSizeOfCubes;

    int *leafParent;
    int *nodeParent;
    int *nodeI, *nodeCount;
    float *nodeValue;
    float *nodeX, *nodeY, *nodeZ;
    float *nodeVX, *nodeVY, *nodeVZ;

    __host__ __device__
    combineMergeTrees(unsigned int *cubeMapping, unsigned int *cubeId, int *tmpNxt, int *tmpFree,
        int *leafParent, int *nodeParent,
        int *nodeI, float *nodeValue, int *nodeCount,
        float *nodeX, float *nodeY, float *nodeZ,
        float *nodeVX, float *nodeVY, float *nodeVZ,
        int *edgesSrc, int *edgesDes, float *edgesWeight,
        int *edgeStartOfCubes, int *edgeSizeOfCubes,
        float min_ll, int sizeP, int numOfCubesOri) :
        cubeMapping(cubeMapping), cubeId(cubeId), tmpNxt(tmpNxt), tmpFree(tmpFree),
        leafParent(leafParent), nodeParent(nodeParent),
        nodeI(nodeI), nodeValue(nodeValue), nodeCount(nodeCount),
        nodeX(nodeX), nodeY(nodeY), nodeZ(nodeZ),
        nodeVX(nodeVX), nodeVY(nodeVY), nodeVZ(nodeVZ),
        edgesSrc(edgesSrc), edgesDes(edgesDes), edgesWeight(edgesWeight),
        edgeStartOfCubes(edgeStartOfCubes), edgeSizeOfCubes(edgeSizeOfCubes),
        min_ll(min_ll), sizeP(sizeP), numOfCubesOri(numOfCubesOri) {}

    __host__ __device__
    void operator()(int i)
    {
      int cubeStart  = sizeP*i;
      int cubeEnd    = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

      int cubeStartM = cubeMapping[cubeStart];
      int cubeEndM   = cubeMapping[cubeEnd-1];

      // get the edges
      for(int k=cubeStart; k<cubeEnd; k++)
      {
        int size = 0;
        for(int j=edgeStartOfCubes[k]; j<edgeStartOfCubes[k]+edgeSizeOfCubes[k]; j++)
        {
          int eSrc = edgesSrc[j];
          int eDes = edgesDes[j];

          float eWeight = edgesWeight[j];

          if(!(cubeId[eSrc]>=cubeStartM && cubeId[eSrc]<=cubeEndM) ||
             !(cubeId[eDes]>=cubeStartM && cubeId[eDes]<=cubeEndM))
          {
            edgesSrc[edgeStartOfCubes[k] + size] = eSrc;
            edgesDes[edgeStartOfCubes[k] + size] = eDes;
            edgesWeight[edgeStartOfCubes[k] + size] = eWeight;
            size++;
            continue;
          }

          // use this edge (e), to combine the merge trees
          //-----------------------------------------------------

          float weight = (eWeight < min_ll) ? min_ll : eWeight;

          // find the src & des nodes just below the required weight
          int src = leafParent[eSrc]; while(nodeParent[src]!=-1 && nodeValue[nodeParent[src]]<=weight) src = nodeParent[src];
          int des = leafParent[eDes]; while(nodeParent[des]!=-1 && nodeValue[nodeParent[des]]<=weight) des = nodeParent[des];

          // if src & des already have the same halo id, do NOT do anything
          if(nodeI[src]==nodeI[des]) continue;

          int srcCount = nodeCount[src];
          int desCount = nodeCount[des];

          float srcX = nodeX[src]; float desX = nodeX[des]; float srcVX = nodeVX[src]; float desVX = nodeVX[des];
          float srcY = nodeY[src]; float desY = nodeY[des]; float srcVY = nodeVY[src]; float desVY = nodeVY[des];
          float srcZ = nodeZ[src]; float desZ = nodeZ[des]; float srcVZ = nodeVZ[src]; float desVZ = nodeVZ[des];

          // get the original parents of src & des nodes
          int srcTmp = nodeParent[src];
          int desTmp = nodeParent[des];

          // remove the src & des from their parents
          nodeParent[src] = -1;
          nodeParent[des] = -1;



          // set n node
          int n;
          if(nodeValue[src]==weight && nodeValue[des]==weight) // merge src & des, then fix the loop
          { n = src; nodeParent[des] = n; }
          else if(nodeValue[src]==weight) // set des node's parent to be src, set n to src, then fix the loop
          { n = src; nodeParent[des] = n; }
          else if(nodeValue[des]==weight) // set src node's parent to be des, set n to des, then fix the loop
          { n = des; nodeParent[src] = n; }
          else if(nodeValue[src]!=weight && nodeValue[des]!=weight) // create a new node, set it as parent of both src & des, then fix the loop
          {
            if(tmpNxt[cubeStart]!=-1)
            {
              n = tmpNxt[cubeStart];
              int tmpVal = tmpFree[tmpNxt[cubeStart]];
              tmpFree[tmpNxt[cubeStart]] = -2;
              tmpNxt[cubeStart] = tmpVal;

              nodeParent[src] = n;  nodeParent[des] = n;
            }
            #if THRUST_DEVICE_BACKEND != THRUST_DEVICE_BACKEND_CUDA
            else
              std::cout << "***no Free item .... this shouldn't happen*** " << cubeStart << std::endl;
            #endif
          }

          // set values for the n node
          nodeValue[n] = weight;
          nodeCount[n] = nodeCount[src] + nodeCount[des];
          nodeX[n] = nodeX[src]+nodeX[des];   nodeVX[n] = nodeVX[src]+nodeVX[des];
          nodeY[n] = nodeY[src]+nodeY[des];   nodeVY[n] = nodeVY[src]+nodeVY[des];
          nodeZ[n] = nodeZ[src]+nodeZ[des];   nodeVZ[n] = nodeVZ[src]+nodeVZ[des];
          nodeI[n] = (nodeI[src] < nodeI[des]) ? nodeI[src] : nodeI[des];

          //fix the loop
          while(srcTmp!=-1 && desTmp!=-1)
          {
            if(nodeValue[srcTmp] < nodeValue[desTmp])
            {
              nodeParent[n] = srcTmp;
              nodeI[srcTmp] = (nodeI[srcTmp]<nodeI[n]) ? nodeI[srcTmp] : nodeI[n];
              srcCount = nodeCount[srcTmp];
              nodeCount[srcTmp] += desCount;
              srcX = nodeX[srcTmp];   srcVX = nodeVX[srcTmp];
              srcY = nodeY[srcTmp];   srcVY = nodeVY[srcTmp];
              srcZ = nodeZ[srcTmp];   srcVZ = nodeVZ[srcTmp];
              nodeX[srcTmp] += desX;  nodeVX[srcTmp] += desVX;
              nodeY[srcTmp] += desY;  nodeVY[srcTmp] += desVY;
              nodeZ[srcTmp] += desZ;  nodeVZ[srcTmp] += desVZ;

              n = srcTmp;
              srcTmp = nodeParent[srcTmp];

              nodeParent[n] = -1;
            }
            else if(nodeValue[srcTmp] > nodeValue[desTmp])
            {
              nodeParent[n] = desTmp;
              nodeI[desTmp] = (nodeI[desTmp]<nodeI[n]) ? nodeI[desTmp] : nodeI[n];
              desCount = nodeCount[desTmp];
              nodeCount[desTmp] += srcCount;
              desX = nodeX[desTmp];   desVX = nodeVX[desTmp];
              desY = nodeY[desTmp];   desVY = nodeVY[desTmp];
              desZ = nodeZ[desTmp];   desVZ = nodeVZ[desTmp];
              nodeX[desTmp] += srcX;  nodeVX[desTmp] += srcVX;
              nodeY[desTmp] += srcY;  nodeVY[desTmp] += srcVY;
              nodeZ[desTmp] += srcZ;  nodeVZ[desTmp] += srcVZ;

              n = desTmp;
              desTmp = nodeParent[desTmp];

              nodeParent[n] = -1;
            }
            else if(nodeValue[srcTmp] == nodeValue[desTmp])
            {
              while(nodeParent[srcTmp]!=-1 && nodeValue[srcTmp]==nodeValue[nodeParent[srcTmp]]) srcTmp = nodeParent[srcTmp];
              while(nodeParent[desTmp]!=-1 && nodeValue[desTmp]==nodeValue[nodeParent[desTmp]]) desTmp = nodeParent[desTmp];

              nodeParent[n] = srcTmp;

              if(nodeI[srcTmp] != nodeI[desTmp]) // combine srcTmp & desTmp
              {
                int srcParentTmp = nodeParent[srcTmp];
                int desParentTmp = nodeParent[desTmp];

                nodeParent[desTmp] = srcTmp;

                nodeI[srcTmp] = (nodeI[srcTmp]<nodeI[desTmp]) ? nodeI[srcTmp] : nodeI[desTmp];
                nodeCount[srcTmp] += nodeCount[desTmp];
                nodeX[srcTmp] += nodeX[desTmp];  nodeVX[srcTmp] += nodeVX[desTmp];
                nodeY[srcTmp] += nodeY[desTmp];  nodeVY[srcTmp] += nodeVY[desTmp];
                nodeZ[srcTmp] += nodeZ[desTmp];  nodeVZ[srcTmp] += nodeVZ[desTmp];

                n = srcTmp;
                srcTmp = srcParentTmp;
                desTmp = desParentTmp;

                nodeParent[n] = -1;
              }
              else
              {
                srcTmp = -1;  desTmp = -1;
                break;
              }
            }
          }



          if(srcTmp!=-1) nodeParent[n] = srcTmp;
          while(srcTmp!=-1)
          {
            nodeI[srcTmp] = (nodeI[srcTmp]<nodeI[n]) ? nodeI[srcTmp] : nodeI[n];
            srcCount = nodeCount[srcTmp];
            nodeCount[srcTmp] += desCount;
            srcX = nodeX[srcTmp];   srcVX = nodeVX[srcTmp];
            srcY = nodeY[srcTmp];   srcVY = nodeVY[srcTmp];
            srcZ = nodeZ[srcTmp];   srcVZ = nodeVZ[srcTmp];
            nodeX[srcTmp] += desX;  nodeVX[srcTmp] += desVX;
            nodeY[srcTmp] += desY;  nodeVY[srcTmp] += desVY;
            nodeZ[srcTmp] += desZ;  nodeVZ[srcTmp] += desVZ;

            n = srcTmp;
            srcTmp = nodeParent[srcTmp];
          }


          if(desTmp!=-1) nodeParent[n] = desTmp;
          while(desTmp!=-1)
          {
            nodeI[desTmp] = (nodeI[desTmp]<nodeI[n]) ? nodeI[desTmp] : nodeI[n];
            desCount = nodeCount[desTmp];
            nodeCount[desTmp] += srcCount;
            desX = nodeX[desTmp];   desVX = nodeVX[desTmp];
            desY = nodeY[desTmp];   desVY = nodeVY[desTmp];
            desZ = nodeZ[desTmp];   desVZ = nodeVZ[desTmp];
            nodeX[desTmp] += srcX;  nodeVX[desTmp] += srcVX;
            nodeY[desTmp] += srcY;  nodeVY[desTmp] += srcVY;
            nodeZ[desTmp] += srcZ;  nodeVZ[desTmp] += srcVZ;

            n = desTmp;
            desTmp = nodeParent[desTmp];
          }

          //-----------------------------------------------------

        }

        edgeSizeOfCubes[k] = size;
      }
    }
  };

  // jump leaf's parent pointers
  struct jumpLeafPointers : public thrust::unary_function<int, void>
  {
    int   *leafParent, *nodeParent;
    int   *nodeCount;
    float *nodeValue;

    __host__ __device__
    jumpLeafPointers(int *leafParent, int *nodeParent, int *nodeCount, float *nodeValue) :
      leafParent(leafParent), nodeParent(nodeParent), nodeCount(nodeCount), nodeValue(nodeValue) {}

    __host__ __device__
    bool operator()(int i)
    {
      int tmp = leafParent[i];

      while(tmp!=-1 && nodeParent[tmp]!=-1 && nodeValue[tmp]==nodeValue[nodeParent[tmp]])
      {
        leafParent[i] = nodeParent[tmp]; //jump a pointer
        tmp = nodeParent[tmp]; // continue
      }
    }
  };

  // jump node's parent pointers
  struct jumpNodePointers : public thrust::unary_function<int, void>
  {
    int   *nodeParent;
    int   *nodeCount;
    float *nodeValue;

    __host__ __device__
    jumpNodePointers(int *nodeParent, int *nodeCount, float *nodeValue) :
      nodeParent(nodeParent), nodeCount(nodeCount), nodeValue(nodeValue) {}

    __host__ __device__
    bool operator()(int i)
    {
      int tmp = nodeParent[i];

      while(tmp!=-1 && nodeParent[tmp]!=-1 && nodeValue[tmp]==nodeValue[nodeParent[tmp]])
      {
        nodeParent[i] = nodeParent[tmp]; //jump a pointer
        tmp = nodeParent[tmp]; // continue
      }
    }
  };



  //------- output results

  // print cube details from device vectors
  void outputCubeDetails(std::string title)
  {
    std::cout << title << std::endl << std::endl;

    std::cout << "-- Dim    (" << lBoundX << "," << lBoundY << "," << lBoundZ << "), (" << uBoundX << "," << uBoundY << "," << uBoundZ << ")" << std::endl;
    std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;
    std::cout << std::endl << "----------------------" << std::endl << std::endl;

    std::cout << "cubeID    "; thrust::copy(cubeId.begin(), cubeId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
    std::cout << std::endl;
    std::cout << "startOfCube "; thrust::copy(particleStartOfCubes.begin(), particleStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
    std::cout << "----------------------" << std::endl << std::endl;
  }

};

}

#endif


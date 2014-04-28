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

  float  totalTime;                                               // total time taken for halo finding

  double max_ll, min_ll;                                          // maximum & minimum linking lengths

  int k;                                                          // k-way merge for global step
  unsigned long long mergetreeSize;                               // total size of the global merge tree
  unsigned long long side, size, ite;                             // variable needed to determine the neighborhood cubes

  float cubeLen;                                                  // length of the cube
  unsigned long long cubes;                                       // total number of cubes which should be considered, usually this is equal to cubesNonEmpty
  unsigned long long numOfCubes;                                  // total number of cubes in space
  unsigned long long cubesInX, cubesInY, cubesInZ;                // number of cubes in each dimension

  unsigned long long maxChunkSize, chunks;                        // maxChunkSize & number of chunks which should be considered
  thrust::device_vector<unsigned long long> sizeOfChunks;         // size of each chunk of cubes
  thrust::device_vector<unsigned long long> startOfChunks;        // start of each chunk of cubes

  unsigned long long binSize, bins;                               // binSize & number of bins for grouping the cubes
  thrust::device_vector<unsigned long long> binStart;             // start cubeMapping id for each bin

  thrust::device_vector<unsigned long long> idOriginal;           // for each particle, original id
  thrust::device_vector<unsigned long long> cubeId;               // for each particle, cube id

  thrust::device_vector<unsigned long long> cubeMapping;          // array with only nonempty cube ids
  thrust::device_vector<unsigned long long> particleStartOfCubes; // stratInd of cubes (only nonempty cube ids)

  unsigned long long numOfEdges;                                  // total number of edges in space

  thrust::device_vector<unsigned long long> edgesSrc, edgesDes;   // edge of cubes - src & des
  thrust::device_vector<float> edgesWeight;                       // edge of cubes - weight
  thrust::device_vector<int>   edgeSizeOfCubes;                   // size of edges in cubes
  thrust::device_vector<int>   edgeStartOfCubes;                  // start of edges in cubes

  thrust::device_vector<long long> tmpNxt, tmpEnd, tmpFree;       // stores details of free items in merge tree

  // stores parent & parentS details for leafs in the merge tree
  thrust::device_vector<int>   leafParent, leafParentS;           // parent & parentS of each leaf

  // stores all details for nodes in the merge tree
  thrust::device_vector<int>   nodeI;                             // index of each node
  thrust::device_vector<int>   nodeM;                             // index of each node
  thrust::device_vector<int>   nodeCount;                         // size of each node
  thrust::device_vector<int>   nodeParent;                        // parent of each node
  thrust::device_vector<float> nodeValue;                         // function value of each node
  thrust::device_vector<float> nodeX, nodeY, nodeZ;               // positions of each node
  thrust::device_vector<float> nodeVX, nodeVY, nodeVZ;            // velocities of each node

  halo_merge(float min_linkLength, float max_linkLength, int k = 2, std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1) : halo(filename, format, n, np, rL)
  {
    if(numOfParticles!=0)
    {
      // initializations

      struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff1, diff2, diff3, diff4;

      // un-normalize linkLengths so that it will work with box size distances
      min_ll  = min_linkLength*xscal; // get min_linkinglength
      max_ll  = max_linkLength*xscal; // get max_linkinglength
      cubeLen = min_ll / std::sqrt(3); // min_ll*min_ll = 3*cubeLen*cubeLen

      this->k = k;

      if(cubeLen <= 0) { std::cout << "--ERROR : please specify a valid cubeLen... current cubeLen is " << cubeLen/xscal << std::endl; return; }

      // get the number of neighbors which should be checked for each cube
      side = (1 + std::ceil(max_ll/cubeLen)*2);
      size = side*side*side;
      ite  = (size-1)/2;

      // set total number of cubes
      cubesInX = std::ceil((uBoundX-lBoundX)/cubeLen); if (cubesInX==0) cubesInX = 1;
      cubesInY = std::ceil((uBoundY-lBoundY)/cubeLen); if (cubesInY==0) cubesInY = 1;
      cubesInZ = std::ceil((uBoundZ-lBoundZ)/cubeLen); if (cubesInZ==0) cubesInZ = 1;
      numOfCubes = cubesInX*cubesInY*cubesInZ;

      std::cout << "lBoundS " << lBoundX << " " << lBoundY << " " << lBoundZ << " " << std::endl;
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

      std::cout << "-- localStep done" << std::endl << std::endl;

      // merge local merge trees to create the global merge tree
      gettimeofday(&mid5, 0);
      globalStep();
      gettimeofday(&end, 0);

      std::cout << "-- globalStep done" << std::endl << std::endl;

      getSizeOfMergeTree();
      writeMergeTreeToFile(filename);

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

      // sort particles again by their original particle id - for validation purposes
      thrust::sort_by_key(idOriginal.begin(), idOriginal.end(), thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin(), leafVX.begin(), leafVY.begin(), leafVZ.begin(), leafM.begin(), leafI.begin(), leafParent.begin())));

      sizeOfChunks.clear(); startOfChunks.clear();
      binStart.clear();
      cubeId.clear();
      cubeMapping.clear(); particleStartOfCubes.clear();
      edgesSrc.clear();  edgesDes.clear(); edgesWeight.clear(); edgeSizeOfCubes.clear(); edgeStartOfCubes.clear();
      tmpNxt.clear(); tmpEnd.clear(); tmpFree.clear();
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

  // get the size of the merge tree
  void getSizeOfMergeTree()
  {
    mergetreeSize = 2*cubes - thrust::count(nodeCount.begin(), nodeCount.end(), 0);
  }

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

  // for a given node set its halo id, for particles in filtered halos set id to -1
  struct checkIfNode : public thrust::unary_function<int, void>
  {
    int *nodeI, *tmp;

    __host__ __device__
    checkIfNode(int *nodeI, int *tmp) : nodeI(nodeI), tmp(tmp) {}

    __host__ __device__
    void operator()(int i)
    {
      tmp[i] = (nodeI[i]==-1) ? 0 : 1;
    }
  };



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
    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<unsigned long long>::iterator> new_end;

    // find unique halo ids & one particle id which belongs to that halo
    haloIndexUnique.resize(numOfParticles);
    thrust::copy(haloIndex.begin(), haloIndex.end(), haloIndexUnique.begin());
    thrust::sequence(idOriginal.begin(), idOriginal.end());
    thrust::stable_sort_by_key(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfParticles, idOriginal.begin(), thrust::greater<int>());
    new_end = thrust::unique_by_key(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfParticles, idOriginal.begin());

    numOfHalos = thrust::get<0>(new_end) - haloIndexUnique.begin();
    if(haloIndexUnique[numOfHalos-1]==-1) numOfHalos--;

    thrust::reverse(idOriginal.begin(), idOriginal.begin()+numOfHalos);

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
                     thrust::raw_pointer_cast(&*idOriginal.begin()),
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

    unsigned long long *particleId;
    int *haloCount;
    float *haloX, *haloY, *haloZ;
    float *haloVX, *haloVY, *haloVZ;

    __host__ __device__
    setHaloStats(int *leafParentS, int *nodeCount,
      float *nodeX, float *nodeY, float *nodeZ,
      float *nodeVX, float *nodeVY, float *nodeVZ,
      unsigned long long *particleId, int *haloCount,
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
    thrust::device_vector<unsigned long long>::iterator new_end;

    thrust::sequence(idOriginal.begin(), idOriginal.begin()+numOfParticles);

    new_end = thrust::remove_if(idOriginal.begin(), idOriginal.begin()+numOfParticles,
        invalidHalo(thrust::raw_pointer_cast(&*haloIndex.begin())));

    numOfHaloParticles = new_end - idOriginal.begin();

    haloIndex_f.resize(numOfHaloParticles);
    inputX_f.resize(numOfHaloParticles);
    inputY_f.resize(numOfHaloParticles);
    inputZ_f.resize(numOfHaloParticles);

    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfHaloParticles,
      getHaloParticlePositions(thrust::raw_pointer_cast(&*leafX.begin()),
                               thrust::raw_pointer_cast(&*leafY.begin()),
                               thrust::raw_pointer_cast(&*leafZ.begin()),
                               thrust::raw_pointer_cast(&*haloIndex.begin()),
                               thrust::raw_pointer_cast(&*idOriginal.begin()),
                               thrust::raw_pointer_cast(&*inputX_f.begin()),
                               thrust::raw_pointer_cast(&*inputY_f.begin()),
                               thrust::raw_pointer_cast(&*inputZ_f.begin()),
                               thrust::raw_pointer_cast(&*haloIndex_f.begin())));
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
    unsigned long long *particleId;

    float *leafX, *leafY, *leafZ;
    float *inputX_f, *inputY_f, *inputZ_f;

    int *haloIndex, *haloIndex_f;

    __host__ __device__
    getHaloParticlePositions(float *leafX, float *leafY, float *leafZ, int *haloIndex, unsigned long long *particleId,
      float *inputX_f, float *inputY_f, float *inputZ_f, int *haloIndex_f) :
      leafX(leafX), leafY(leafY), leafZ(leafZ), haloIndex(haloIndex), particleId(particleId),
      inputX_f(inputX_f), inputY_f(inputY_f), inputZ_f(inputZ_f), haloIndex_f(haloIndex_f) {}

    __host__ __device__
    void operator()(int i)
    {
      int n = particleId[i];

      inputX_f[i] = leafX[n];
      inputY_f[i] = leafY[n];
      inputZ_f[i] = leafZ[n];

      haloIndex_f[i] = haloIndex[n];
    }
  };



  //------- initialization step

  // divide space into cubes
  void initCubes()
  {
    struct timeval begin, mid1, mid2, mid3, end, diff1, diff2, diff3, diff4;

    gettimeofday(&begin, 0);

    // compute which cube each particle is in
    cubeId.resize(numOfParticles);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
        setCubeIdOfParticle(thrust::raw_pointer_cast(&*leafX.begin()),
                            thrust::raw_pointer_cast(&*leafY.begin()),
                            thrust::raw_pointer_cast(&*leafZ.begin()),
                            thrust::raw_pointer_cast(&*cubeId.begin()),
                            cubeLen, lBoundX, lBoundY, lBoundZ, cubesInX, cubesInY, cubesInZ));

    gettimeofday(&mid1, 0);

    // sort particles by cube
    idOriginal.resize(numOfParticles);
    thrust::sequence(idOriginal.begin(), idOriginal.end(), 0);
    thrust::sort_by_key(cubeId.begin(), cubeId.end(), thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin(), leafVX.begin(), leafVY.begin(), leafVZ.begin(), leafM.begin(), leafI.begin(), idOriginal.begin())));

    gettimeofday(&mid2, 0);

    // get the size,start & cube mapping details for only non empty cubes
    unsigned long long num = (numOfParticles<numOfCubes) ? numOfParticles : numOfCubes;
    cubeMapping.resize(num);
    particleStartOfCubes.resize(num);
    thrust::pair<thrust::device_vector<unsigned long long>::iterator, thrust::device_vector<unsigned long long>::iterator> new_end;
    new_end = thrust::reduce_by_key(cubeId.begin(), cubeId.end(), ConstantIterator(1), cubeMapping.begin(), particleStartOfCubes.begin());
    unsigned long long cubesNonEmpty = thrust::get<0>(new_end) - cubeMapping.begin();
    unsigned long long cubesEmpty    = (numOfCubes - cubesNonEmpty);
    cubes = cubesNonEmpty; // get the cubes which should be considered

    gettimeofday(&mid3, 0);

    //setChunks(); //set chunks for load balance
    thrust::exclusive_scan(particleStartOfCubes.begin(), particleStartOfCubes.begin()+cubes, particleStartOfCubes.begin());
    cubeMapping.resize(cubes);
    particleStartOfCubes.resize(cubes);

    gettimeofday(&end, 0);

    std::cout << cubesEmpty << " of " << numOfCubes << " cubes are empty. (" << (((double)cubesEmpty*100)/(double)numOfCubes) << "%) ... non empty cubes " << cubesNonEmpty << std::endl;

    std::cout << std::endl;
    std::cout << "'initCubes' Time division:" << std::endl << std::flush;
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed0: " << seconds1 << " s for setCubeIdOfParticle" << std::endl << std::flush;
    timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed1: " << seconds2 << " s for sort particles by cube" << std::endl << std::flush;
    timersub(&mid3, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed2: " << seconds3 << " s for reduce_by_key" << std::endl << std::flush;
    timersub(&end, &mid3, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed3: " << seconds4 << " s for setChunks, exclusive_scan & resize" << std::endl << std::flush;

    #ifdef TEST
      outputCubeDetails("cube details"); // output cube details
    #endif
  }

  // for a given particle, set its cube id
  struct setCubeIdOfParticle : public thrust::unary_function<int, void>
  {
    float cubeLen;
    unsigned long long *cubeId;
    unsigned long long  cubesInX, cubesInY, cubesInZ;

    float  lBoundX, lBoundY, lBoundZ;
    float *leafX, *leafY, *leafZ;

    __host__ __device__
    setCubeIdOfParticle(float *leafX, float *leafY, float *leafZ,
      unsigned long long *cubeId, float cubeLen,
      float lBoundX, float lBoundY, float lBoundZ,
      unsigned long long cubesInX, unsigned long long cubesInY, unsigned long long cubesInZ) :
      leafX(leafX), leafY(leafY), leafZ(leafZ),
      cubeId(cubeId), cubeLen(cubeLen),
      lBoundX(lBoundX), lBoundY(lBoundY), lBoundZ(lBoundZ),
      cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ){}

    __host__ __device__
    void operator()(int i)
    {
      // get x,y,z coordinates for the cube
      unsigned long long x = (leafX[i]-lBoundX)/cubeLen; if(x>=cubesInX) x = cubesInX-1;
      unsigned long long y = (leafY[i]-lBoundY)/cubeLen; if(y>=cubesInY) y = cubesInY-1;
      unsigned long long z = (leafZ[i]-lBoundZ)/cubeLen; if(z>=cubesInZ) z = cubesInZ-1;

      cubeId[i] = (unsigned long long)(x + (y*cubesInX) + (z*cubesInX*cubesInY)); // get cube id
    }
  };

  // group the set of cubes to chunks of same computation sizes (for load balance)
  void setChunks()
  {
    sizeOfChunks.resize(cubes);
    startOfChunks.resize(cubes);

    maxChunkSize = *(thrust::max_element(particleStartOfCubes.begin(), particleStartOfCubes.begin()+cubes));

    thrust::inclusive_scan(particleStartOfCubes.begin(), particleStartOfCubes.begin()+cubes, sizeOfChunks.begin());

    thrust::fill(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
    thrust::copy_if(CountingIterator(0), CountingIterator(0)+cubes, startOfChunks.begin(),
       isStartOfChunks(thrust::raw_pointer_cast(&*sizeOfChunks.begin()), maxChunkSize));

    chunks = cubes - thrust::count(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);

    sizeOfChunks.clear();
    startOfChunks.resize(chunks);

    std::cout << "maxChunkSize " << maxChunkSize << " chunks " << chunks << std::endl;
  }

  // check whether this cube is the start of a chunk
  struct isStartOfChunks : public thrust::unary_function<int, void>
  {
    unsigned long long maxChunkSize;
    unsigned long long *sizeOfChunks;

    __host__ __device__
    isStartOfChunks(unsigned long long *sizeOfChunks, unsigned long long maxChunkSize) :
      sizeOfChunks(sizeOfChunks), maxChunkSize(maxChunkSize) {}

    __host__ __device__
    bool operator()(int i)
    {
      if(i==0) return true;

      int a = sizeOfChunks[i] / maxChunkSize;  int b = sizeOfChunks[i-1] / maxChunkSize;
      int d = sizeOfChunks[i] % maxChunkSize;  int e = sizeOfChunks[i-1] % maxChunkSize;

      if((a!=b && d==0 && e==0) || (a!=b && d!=0) || (a==b && e==0)) return true;

      return false;
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

    thrust::fill(nodeCount.begin(),  nodeCount.end(),   0);
    thrust::fill(nodeParent.begin(), nodeParent.end(), -1);
  }



  //------- local step

  // locally, create the local merge trees for each cube
  void localStep()
  {
    struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff1, diff2, diff3, diff4, diff5, diff6;

    gettimeofday(&begin, 0);

    // set bins
    binSize = cubesInX;
    bins = cubesInY*cubesInZ;
    binStart.resize(bins);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        setBins(thrust::raw_pointer_cast(&*cubeMapping.begin()),
                thrust::raw_pointer_cast(&*binStart.begin()),
                binSize));
    thrust::inclusive_scan(binStart.begin(), binStart.end(), binStart.begin(), thrust::maximum<unsigned long long>());
    std::cout << std::endl << "bins " << bins << " binSize " << binSize << std::endl;

    std::cout << "----binStart "; thrust::copy(binStart.begin(), binStart.begin()+2, std::ostream_iterator<unsigned long long>(std::cout, " ")); std::cout << std::endl;

    gettimeofday(&mid1, 0);

    // for each cube, get neighbor details
    edgeSizeOfCubes.resize(cubes);
    edgeStartOfCubes.resize(cubes);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,//chunks,
        getNeighborDetails(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
                           thrust::raw_pointer_cast(&*startOfChunks.begin()),
                           thrust::raw_pointer_cast(&*cubeMapping.begin()),
                           thrust::raw_pointer_cast(&*binStart.begin()),
                           thrust::raw_pointer_cast(&*leafX.begin()),
                           thrust::raw_pointer_cast(&*leafY.begin()),
                           thrust::raw_pointer_cast(&*leafZ.begin()),
                           max_ll, side, cubesInX, cubesInY, cubesInZ, cubes, chunks, binSize, bins, numOfParticles,
                           thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin())));

    std::cout << "----edgeSizeOfCubes "; thrust::copy(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+2, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;

    gettimeofday(&mid2, 0);

    // for each cube, set the space required for storing edges
    thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end(), edgeStartOfCubes.begin());
    numOfEdges = edgeStartOfCubes[cubes-1] + edgeSizeOfCubes[cubes-1]; // size of edges array

    // init edge arrays
    edgesSrc.resize(numOfEdges);
    edgesDes.resize(numOfEdges);
    edgesWeight.resize(numOfEdges);
    std::cout << std::endl << "numOfEdges " << numOfEdges << std::endl; // set the correct number of edges

    std::cout << "----edgeSizeOfCubes "; thrust::copy(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+2, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;

    gettimeofday(&mid3, 0);

    // for each cube, get the set of edges
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,//chunks,
        getEdges(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
                 thrust::raw_pointer_cast(&*startOfChunks.begin()),
                 thrust::raw_pointer_cast(&*cubeMapping.begin()),
                 thrust::raw_pointer_cast(&*binStart.begin()),
                 thrust::raw_pointer_cast(&*leafX.begin()),
                 thrust::raw_pointer_cast(&*leafY.begin()),
                 thrust::raw_pointer_cast(&*leafZ.begin()),
                 max_ll, min_ll, side, cubesInX, cubesInY, cubesInZ, cubes, chunks, binSize, bins, numOfParticles,
                 thrust::raw_pointer_cast(&*edgesSrc.begin()),
                 thrust::raw_pointer_cast(&*edgesDes.begin()),
                 thrust::raw_pointer_cast(&*edgesWeight.begin()),
                 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));

    numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end());
    std::cout << std::endl << "numOfEdges " << numOfEdges << std::endl; // set the correct number of edges

    std::cout << "----edgesWeight "; thrust::copy(edgesWeight.begin(), edgesWeight.begin()+2, std::ostream_iterator<float>(std::cout, " ")); std::cout << std::endl;

    gettimeofday(&mid4, 0);

    // set vectors necessary for merge tree construction
    tmpNxt.resize(cubes);
    tmpEnd.resize(cubes);
    tmpFree.resize(2*cubes);
    thrust::sequence(tmpFree.begin(), tmpFree.end(), 1);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
        initNxt(thrust::raw_pointer_cast(&*tmpNxt.begin()),
                thrust::raw_pointer_cast(&*tmpEnd.begin()),
                thrust::raw_pointer_cast(&*tmpFree.begin())));

    std::cout << "----tmpNxt "; thrust::copy(tmpNxt.begin(), tmpNxt.begin()+2, std::ostream_iterator<long long>(std::cout, " ")); std::cout << std::endl;

    gettimeofday(&mid5, 0);

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
                           thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
                           thrust::raw_pointer_cast(&*tmpNxt.begin()),
                           thrust::raw_pointer_cast(&*tmpFree.begin()),
                           min_ll, numOfParticles, cubes));

    std::cout << "----nodeI "; thrust::copy(nodeI.begin(), nodeI.begin()+2, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl;

    gettimeofday(&end, 0);

    binStart.clear();
    startOfChunks.clear();

    std::cout << std::endl;
    std::cout << "'localStep' Time division:" << std::endl << std::flush;
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed0: " << seconds1 << " s for initBins" << std::endl << std::flush;
    timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed1: " << seconds2 << " s for getNeighborDetails" << std::endl << std::flush;
    timersub(&mid3, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed2: " << seconds3 << " s for initEdgeArrays" << std::endl << std::flush;
    timersub(&mid4, &mid3, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed3: " << seconds4 << " s for getEdges" << std::endl << std::flush;
    timersub(&mid5, &mid4, &diff5);
    float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
    std::cout << "Time elapsed4: " << seconds5 << " s for initNxt" << std::endl << std::flush;
    timersub(&end, &mid5, &diff6);
    float seconds6 = diff6.tv_sec + 1.0E-6*diff6.tv_usec;
    std::cout << "Time elapsed5: " << seconds6 << " s for createSubMergeTree" << std::endl << std::flush;
  }

  // for each bin, set the cubeStart in cubeMapping
  struct setBins : public thrust::unary_function<int, void>
  {
    unsigned long long  binSize;
    unsigned long long *binStart;
    unsigned long long *cubeMapping;

    __host__ __device__
    setBins(unsigned long long *cubeMapping, unsigned long long *binStart, unsigned long long binSize) :
      cubeMapping(cubeMapping), binStart(binStart), binSize(binSize) {}

    __host__ __device__
    void operator()(int i)
    {
      unsigned long long m = (i!=0) ? cubeMapping[i-1]/binSize : -1;
      unsigned long long n = cubeMapping[i]/binSize;

      if(m!=n) binStart[m+1] = i;
    }
  };

  //for each cube, sum the number of particles in neighbor cubes & get the sum of non empty neighbor cubes
  struct getNeighborDetails : public thrust::unary_function<int, void>
  {
    int    numOfParticles;
    int    side;
    float  max_ll;
    float *leafX, *leafY, *leafZ;

    unsigned long long  chunks;
    unsigned long long *startOfChunks;
    unsigned long long *cubeMapping;
    unsigned long long *particleStartOfCubes;

    unsigned long long  binSize, bins;
    unsigned long long *binStart;

    unsigned long long  cubesInX, cubesInY, cubesInZ, cubes;

    int *edgeSizeOfCubes;

    __host__ __device__
    getNeighborDetails(unsigned long long *particleStartOfCubes, unsigned long long *startOfChunks,
        unsigned long long *cubeMapping, unsigned long long *binStart,
        float *leafX, float *leafY, float *leafZ,
        float max_ll, int side,
        unsigned long long cubesInX, unsigned long long cubesInY, unsigned long long cubesInZ, unsigned long long cubes,
        unsigned long long chunks, unsigned long long binSize, unsigned long long bins, int numOfParticles,
        int *edgeSizeOfCubes) :
        particleStartOfCubes(particleStartOfCubes), startOfChunks(startOfChunks),
        cubeMapping(cubeMapping), binStart(binStart),
        leafX(leafX), leafY(leafY), leafZ(leafZ),
        max_ll(max_ll), side(side),
        cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), cubes(cubes),
        chunks(chunks), binSize(binSize), bins(bins), numOfParticles(numOfParticles),
        edgeSizeOfCubes(edgeSizeOfCubes) {}

    __host__ __device__
    void operator()(int i)
    {
      unsigned long long l = i;
//      unsigned long long chunkSize = (i+1<chunks) ? startOfChunks[i+1]-startOfChunks[i] : cubes-startOfChunks[i];
//      for(unsigned long long l=startOfChunks[i]; l<startOfChunks[i]+chunkSize; l++)
      {
        unsigned long long i_mapped = cubeMapping[l];

        // get x,y,z coordinates for the cube
        const int x =  i_mapped % cubesInX;
        const int y = (i_mapped / cubesInX) % cubesInY;
        const int z =  i_mapped / (cubesInX*cubesInY);

        int len = (side-1)/2;

        int sumNonEmptyCubes = 0;
        for(int currentZ=z-len; currentZ<=z+len; currentZ++)
        {
          for(int currentY=y-len; currentY<=y+len; currentY++)
          {
            for(int currentX=x-len; currentX<=x+len; currentX++)
            {
              if(currentX==x && currentY==y && currentZ==z) goto loopEnd;

              if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
              {
                unsigned long long cube_mapped = currentX  + currentY*cubesInX + currentZ*(cubesInY*cubesInX);

                int bin = cube_mapped/binSize;
                int from = binStart[bin];
                int to = (bin+1 < bins) ? binStart[bin+1]-1 : cubes;
                long long cube = -1;
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
                if(cube==-1) continue;

                int size1 = (l+1<cubes) ? particleStartOfCubes[l+1]-particleStartOfCubes[l] : numOfParticles-particleStartOfCubes[l];
                int size2 = (cube+1<cubes) ? particleStartOfCubes[cube+1]-particleStartOfCubes[cube] : numOfParticles-particleStartOfCubes[cube];

                // for each particle in this cube
                bool found = false;
                for(int j=particleStartOfCubes[l]; j<particleStartOfCubes[l]+size1; j++)
                {
                  float3 p_j = make_float3(leafX[j], leafY[j], leafZ[j]);

                  // compare with particles in neighboring cube
                  for(int k=particleStartOfCubes[cube]; k<particleStartOfCubes[cube]+size2; k++)
                  {
                    float3 p_k = make_float3(leafX[k], leafY[k], leafZ[k]);

                    double dist = ((p_j.x-p_k.x)*(p_j.x-p_k.x) + (p_j.y-p_k.y)*(p_j.y-p_k.y) + (p_j.z-p_k.z)*(p_j.z-p_k.z));
                    if(dist < max_ll*max_ll)
                    {
                      found = true;
                      goto loop;
                    }
                  }
                }

                // add edge
                loop:
                if(found) sumNonEmptyCubes++;  //sum the non empty neighbor cubes
             }
            }
          }
        }

        loopEnd:
        edgeSizeOfCubes[l] = sumNonEmptyCubes; // store sum of non empty neighbor cubes for this cube
      }
    }
  };

  // for each cube, get the set of edges after comparing
  struct getEdges : public thrust::unary_function<int, void>
  {
    int    numOfParticles;
    int    side;
    float  max_ll, min_ll;
    float *leafX, *leafY, *leafZ;

    unsigned long long  chunks;
    unsigned long long *startOfChunks;
    unsigned long long *cubeMapping;
    unsigned long long *particleStartOfCubes;

    unsigned long long  binSize, bins;
    unsigned long long *binStart;

    unsigned long long  cubesInX, cubesInY, cubesInZ, cubes;

    unsigned long long *edgesSrc, *edgesDes;
    float *edgesWeight;
    int   *edgeStartOfCubes, *edgeSizeOfCubes;

    __host__ __device__
    getEdges(unsigned long long *particleStartOfCubes, unsigned long long *startOfChunks,
        unsigned long long *cubeMapping, unsigned long long *binStart,
        float *leafX, float *leafY, float *leafZ,
        float max_ll, float min_ll, int side,
        unsigned long long cubesInX, unsigned long long cubesInY, unsigned long long cubesInZ,
        unsigned long long cubes, unsigned long long chunks,
        unsigned long long binSize, unsigned long long bins, int numOfParticles,
        unsigned long long *edgesSrc, unsigned long long *edgesDes, float *edgesWeight,
        int *edgeSizeOfCubes, int *edgeStartOfCubes) :
        particleStartOfCubes(particleStartOfCubes), startOfChunks(startOfChunks),
        cubeMapping(cubeMapping), binStart(binStart),
        leafX(leafX), leafY(leafY), leafZ(leafZ),
        max_ll(max_ll), min_ll(min_ll), side(side),
        cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ),
        cubes(cubes), chunks(chunks),
        binSize(binSize), bins(bins), numOfParticles(numOfParticles),
        edgesSrc(edgesSrc), edgesDes(edgesDes), edgesWeight(edgesWeight),
        edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

    __host__ __device__
    void operator()(int i)
    {
      unsigned long long l = i;
//      unsigned long long chunkSize = (i+1<chunks) ? startOfChunks[i+1]-startOfChunks[i] : cubes-startOfChunks[i];
//      for(unsigned long long l=startOfChunks[i]; l<startOfChunks[i]+chunkSize; l++)
      {
        unsigned long long i_mapped = cubeMapping[l];

        // get x,y,z coordinates for the cube
        const int x =  i_mapped % cubesInX;
        const int y = (i_mapped / cubesInX) % cubesInY;
        const int z =  i_mapped / (cubesInX*cubesInY);

        int len = (side-1)/2;

        int size = 0;
        int start = edgeStartOfCubes[l];
        for(int currentZ=z-len; currentZ<=z+len; currentZ++)
        {
          for(int currentY=y-len; currentY<=y+len; currentY++)
          {
            for(int currentX=x-len; currentX<=x+len; currentX++)
            {
              if(currentX==x && currentY==y && currentZ==z) goto loopEnd;

              if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
              {
                unsigned long long cube_mapped = currentX  + currentY*cubesInX + currentZ*(cubesInY*cubesInX);

                int bin = cube_mapped/binSize;
                int from = binStart[bin];
                int to = (bin+1 < bins) ? binStart[bin+1]-1 : cubes;
                long long cube = -1;
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
                if(cube==-1) continue;

                int size1 = (l+1<cubes) ? particleStartOfCubes[l+1]-particleStartOfCubes[l] : numOfParticles-particleStartOfCubes[l];
                int size2 = (cube+1<cubes) ? particleStartOfCubes[cube+1]-particleStartOfCubes[cube] : numOfParticles-particleStartOfCubes[cube];

                // for each particle in this cube
                int eSrc, eDes;
                float eWeight = max_ll*max_ll;
                for(int j=particleStartOfCubes[l]; j<particleStartOfCubes[l]+size1; j++)
                {
                  float3 p_j = make_float3(leafX[j], leafY[j], leafZ[j]);

                  // compare with particles in neighboring cube
                  for(int k=particleStartOfCubes[cube]; k<particleStartOfCubes[cube]+size2; k++)
                  {
                    float3 p_k = make_float3(leafX[k], leafY[k], leafZ[k]);

                    double dist = ((p_j.x-p_k.x)*(p_j.x-p_k.x) + (p_j.y-p_k.y)*(p_j.y-p_k.y) + (p_j.z-p_k.z)*(p_j.z-p_k.z));
                    if(dist < eWeight)
                    {
                      eSrc = j; eDes = k; eWeight = dist;
                      if(eWeight < min_ll*min_ll) goto loop;
                    }
                  }
                }

                // add edge
                loop:
                if(eWeight < max_ll*max_ll)
                {
                  edgesSrc[start + size] = eSrc;
                  edgesDes[start + size] = eDes;
                  edgesWeight[start + size] = std::sqrt(eWeight);
                  size++;
                }
              }
            }
          }
        }

        loopEnd:
        edgeSizeOfCubes[l] = size;
      }
    }
  };

  // finalize the init of tmpFree & tmpNxt arrays
  struct initNxt : public thrust::unary_function<int, void>
  {
    long long *tmpNxt, *tmpEnd, *tmpFree;

    __host__ __device__
    initNxt(long long *tmpNxt, long long *tmpEnd, long long *tmpFree) : tmpNxt(tmpNxt), tmpEnd(tmpEnd), tmpFree(tmpFree) {}

    __host__ __device__
    void operator()(int i)
    {
      tmpNxt[i] = 2*i;
      tmpEnd[i] = 2*i+1;
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

    unsigned long long *particleStartOfCubes;
    long long *tmpNxt, *tmpFree;

    float min_ll;
    int numOfParticles;
    unsigned long long cubes;

    __host__ __device__
    createSubMergeTree(int *leafParent, int *leafI,
      float *leafX, float *leafY, float *leafZ,
      float *leafVX, float *leafVY, float *leafVZ,
      int *nodeI, int *nodeCount, float *nodeValue,
      float *nodeX, float *nodeY, float *nodeZ,
      float *nodeVX, float *nodeVY, float *nodeVZ,
      unsigned long long *particleStartOfCubes, long long *tmpNxt, long long *tmpFree,
      float min_ll, int numOfParticles, unsigned long long cubes) :
      leafParent(leafParent), leafI(leafI),
      leafX(leafX), leafY(leafY), leafZ(leafZ),
      leafVX(leafVX), leafVY(leafVY), leafVZ(leafVZ),
      nodeI(nodeI), nodeCount(nodeCount), nodeValue(nodeValue),
      nodeX(nodeX), nodeY(nodeY), nodeZ(nodeZ),
      nodeVX(nodeVX), nodeVY(nodeVY), nodeVZ(nodeVZ),
      particleStartOfCubes(particleStartOfCubes), tmpNxt(tmpNxt), tmpFree(tmpFree),
      min_ll(min_ll), numOfParticles(numOfParticles), cubes(cubes) {}

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

      int size = (i+1<cubes) ? particleStartOfCubes[i+1]-particleStartOfCubes[i] : numOfParticles-particleStartOfCubes[i];

      long long minHaloId = -1;
      for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+size; j++)
      {
        leafParent[j] = n; // n is the index in node lists

        minHaloId = (minHaloId==-1) ? leafI[j] : (minHaloId<leafI[j] ? minHaloId : leafI[j]);

        x+=leafX[j]; vx+=leafVX[j];
        y+=leafY[j]; vy+=leafVY[j];
        z+=leafZ[j]; vz+=leafVZ[j];
      }

      nodeI[n] = minHaloId;
      nodeValue[n] = min_ll;
      nodeCount[n] = size;
      nodeX[n] = x;   nodeVX[n] = vx;
      nodeY[n] = y;   nodeVY[n] = vy;
      nodeZ[n] = z;   nodeVZ[n] = vz;
    }
  };



  //------- global step

  // merge local merge trees to create the global merge tree
  void globalStep()
  {
    unsigned long long cubesOri = cubes;
    int sizeP = 1;

    thrust::device_vector<long long>::iterator new_end;
    thrust::pair<thrust::device_vector<long long>::iterator, thrust::device_vector<long long>::iterator> new_end1;

    thrust::device_vector<long long> A(2*cubesOri);
    thrust::device_vector<long long> B(2*cubesOri);
    thrust::device_vector<long long> C,D,E;
    thrust::sequence(A.begin(), A.end());

    // iteratively combine the cubes two at a time
    int i = 0;
    while(true)
    {
      struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff, diff1, diff2, diff3, diff4, diff5, diff6;

      gettimeofday(&begin, 0);

      if(cubes != (int)std::ceil(((double)cubes/k))) // set new number of cubes & sizeP
      {
        sizeP *= k;
        cubes = (int)std::ceil(((double)cubes/k));
        thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
                  combineFreeLists(thrust::raw_pointer_cast(&*tmpNxt.begin()),
                                   thrust::raw_pointer_cast(&*tmpEnd.begin()),
                                   thrust::raw_pointer_cast(&*tmpFree.begin()),
                                   sizeP, k, cubesOri));
      }
      else if(cubes==1) break;

      std::cout << "----tmpNxt "; thrust::copy(tmpNxt.begin(), tmpNxt.begin()+2, std::ostream_iterator<long long>(std::cout, " ")); std::cout << std::endl;

      gettimeofday(&mid1, 0);

      //combine edges
      thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
            combineEdges(thrust::raw_pointer_cast(&*cubeMapping.begin()),
                         thrust::raw_pointer_cast(&*cubeId.begin()),
                         thrust::raw_pointer_cast(&*edgesSrc.begin()),
                         thrust::raw_pointer_cast(&*edgesDes.begin()),
                         thrust::raw_pointer_cast(&*edgesWeight.begin()),
                         thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
                         thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                         sizeP, k, cubesOri));

      std::cout << "----edgesSrc "; thrust::copy(edgesSrc.begin(), edgesSrc.begin()+2, std::ostream_iterator<long long>(std::cout, " ")); std::cout << std::endl;

      gettimeofday(&mid2, 0);

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

      std::cout << "----nodeParent "; thrust::copy(nodeParent.begin(), nodeParent.begin()+2, std::ostream_iterator<int>(std::cout, " ")); std::cout << " ";

      gettimeofday(&mid3, 0);

      thrust::for_each(CountingIterator(0), CountingIterator(0)+2*cubesOri,
          jumpNodePointers(thrust::raw_pointer_cast(&*nodeParent.begin()),
                           thrust::raw_pointer_cast(&*nodeCount.begin()),
                           thrust::raw_pointer_cast(&*nodeValue.begin())));

      std::cout << "----nodeParent "; thrust::copy(nodeParent.begin(), nodeParent.begin()+2, std::ostream_iterator<int>(std::cout, " ")); std::cout << " ";

      gettimeofday(&mid4, 0);

      thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
          jumpLeafPointers(thrust::raw_pointer_cast(&*leafParent.begin()),
                           thrust::raw_pointer_cast(&*nodeParent.begin()),
                           thrust::raw_pointer_cast(&*nodeValue.begin())));

      std::cout << "----leafParent "; thrust::copy(leafParent.begin(), leafParent.begin()+2, std::ostream_iterator<int>(std::cout, " ")); std::cout << " ";

      gettimeofday(&mid5, 0);

      new_end = thrust::remove_copy_if(A.begin(), A.end(), B.begin(), isUsed(thrust::raw_pointer_cast(&*nodeCount.begin())));

      int size1 = new_end-B.begin();

      C.resize(size1); D.resize(size1); E.resize(size1);
      thrust::for_each(CountingIterator(0), CountingIterator(0)+size1,
          setCubeId(thrust::raw_pointer_cast(&*B.begin()),
                    thrust::raw_pointer_cast(&*C.begin()),
                    sizeP));

      new_end1 = thrust::reduce_by_key(C.begin(), C.end(), ConstantIterator(1), D.begin(), E.begin());

      thrust::exclusive_scan(E.begin(), E.end(), E.begin());

      int size2 = thrust::get<0>(new_end1)-D.begin();

      thrust::for_each(CountingIterator(0), CountingIterator(0)+size2,
                freeNodes(thrust::raw_pointer_cast(&*B.begin()),
                          thrust::raw_pointer_cast(&*D.begin()),
                          thrust::raw_pointer_cast(&*E.begin()),
                          thrust::raw_pointer_cast(&*tmpNxt.begin()),
                          thrust::raw_pointer_cast(&*tmpFree.begin()),
                          thrust::raw_pointer_cast(&*nodeParent.begin()),
                          thrust::raw_pointer_cast(&*nodeCount.begin()),
                          size1, size2));

      std::cout << "----tmpNxt "; thrust::copy(tmpNxt.begin(), tmpNxt.begin()+2, std::ostream_iterator<long long>(std::cout, " ")); std::cout << std::endl;

      gettimeofday(&end, 0);

      timersub(&mid1, &begin, &diff1);
      float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
      std::cout << "Time elapsed " << seconds1 << "s combineFreeLists" << std::endl;
      timersub(&mid2, &mid1, &diff2);
      float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
      std::cout << "Time elapsed " << seconds2 << "s combineEdges" << std::endl;
      timersub(&mid3, &mid2, &diff3);
      float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
      std::cout << "Time elapsed " << seconds3 << "s combineMergeTrees" << std::endl;
      timersub(&mid4, &mid3, &diff4);
      float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
      std::cout << "Time elapsed " << seconds4 << "s jumpNodePointers" << std::endl;
      timersub(&mid5, &mid4, &diff5);
      float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
      std::cout << "Time elapsed " << seconds5 << "s jumpLeafPointers" << std::endl;
      timersub(&end, &mid5, &diff6);
      float seconds6 = diff6.tv_sec + 1.0E-6*diff6.tv_usec;
      std::cout << "Time elapsed " << seconds6 << "s freeNodesNew" << std::endl;

      timersub(&end, &begin, &diff);
      float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
      std::cout << "Iteration: " << i << " ... time elapsed " << seconds << "s nonEmptyCubes " << cubes << std::endl;

      i++;
    }

    cubes = cubesOri;
  }

  // combine free nodes lists of each cube at this iteration
  struct combineFreeLists : public thrust::unary_function<int, void>
  {
    int  sizeP, n;
    unsigned long long numOfCubesOri;

    long long *tmpNxt, *tmpEnd, *tmpFree;

    __host__ __device__
    combineFreeLists(long long *tmpNxt, long long *tmpEnd, long long *tmpFree, int sizeP, int n, unsigned long long numOfCubesOri) :
        tmpNxt(tmpNxt), tmpEnd(tmpEnd), tmpFree(tmpFree), sizeP(sizeP), n(n), numOfCubesOri(numOfCubesOri) {}

    __host__ __device__
    void operator()(int i)
    {
      int cubeStart = sizeP*i;
      int cubeEnd   = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

      int k;
      for(k=cubeStart; k<cubeEnd; k+=sizeP/n)
      { if(tmpNxt[k]!=-1) { tmpNxt[cubeStart] = tmpNxt[k];  break; } }

      int nxt;
      while(k<cubeEnd)
      {
        nxt = tmpEnd[k];

        k += sizeP/n;
        if(k<cubeEnd)
        {
          tmpFree[nxt] = tmpNxt[k];
          tmpEnd[cubeStart] = tmpEnd[k];
        }
      }
    }
  };

  //combine edges
  struct combineEdges
  {
    int  sizeP, n;
    unsigned long long numOfCubesOri;

    unsigned long long *cubeId;
    unsigned long long *cubeMapping;

    unsigned long long *edgesSrc, *edgesDes;
    float *edgesWeight;
    int   *edgeStartOfCubes, *edgeSizeOfCubes;

    __host__ __device__
    combineEdges(unsigned long long *cubeMapping, unsigned long long *cubeId,
        unsigned long long *edgesSrc, unsigned long long *edgesDes,
        float *edgesWeight, int *edgeStartOfCubes, int *edgeSizeOfCubes,
        int sizeP, int n, unsigned long long numOfCubesOri) :
        cubeMapping(cubeMapping), cubeId(cubeId),
        edgesSrc(edgesSrc), edgesDes(edgesDes), edgesWeight(edgesWeight),
        edgeStartOfCubes(edgeStartOfCubes), edgeSizeOfCubes(edgeSizeOfCubes),
        sizeP(sizeP), n(n), numOfCubesOri(numOfCubesOri) {}

    __host__ __device__
    void operator()(int i)
    {
      unsigned long long cubeStart  = sizeP*i;
      unsigned long long cubeMiddle = ((sizeP*i+(sizeP/2))<numOfCubesOri) ? sizeP*i+(sizeP/2) : numOfCubesOri-1;
      unsigned long long cubeEnd    = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

      unsigned long long cubeStartM = cubeMapping[cubeStart];
      unsigned long long cubeEndM   = cubeMapping[cubeEnd-1];

      unsigned long long k = edgeStartOfCubes[cubeStart];
      unsigned long long t = edgeStartOfCubes[cubeStart];
      for(int l=cubeStart; l<cubeEnd; l+=sizeP/n)
      {
        for(int j=edgeStartOfCubes[l]; j<edgeStartOfCubes[l]+edgeSizeOfCubes[l]; j++)
        {
          int eSrc = edgesSrc[j];
          int eDes = edgesDes[j];

          float eWeight = edgesWeight[j];

          if(!(cubeId[eSrc]>=cubeStartM && cubeId[eSrc]<=cubeEndM) ||
             !(cubeId[eDes]>=cubeStartM && cubeId[eDes]<=cubeEndM))
          {
            edgesSrc[k]    = edgesSrc[j];
            edgesDes[k]    = edgesDes[j];
            edgesWeight[k] = edgesWeight[j];
          }
          else
          {
            edgesSrc[k]    = edgesSrc[t];
            edgesDes[k]    = edgesDes[t];
            edgesWeight[k] = edgesWeight[t];

            edgesSrc[t]    = eSrc;
            edgesDes[t]    = eDes;
            edgesWeight[t] = eWeight;
            t++;
          }

          k++;
        }
      }

      edgeSizeOfCubes[cubeStart] = k - edgeStartOfCubes[cubeStart];
    }
  };

  //combine merge trees
  struct combineMergeTrees : public thrust::unary_function<int, void>
  {
    float  min_ll;
    int    sizeP;
    unsigned long long numOfCubesOri;

    unsigned long long *cubeId;
    unsigned long long *cubeMapping;

    long long *tmpNxt, *tmpFree;

    unsigned long long *edgesSrc, *edgesDes;
    float *edgesWeight;
    int   *edgeStartOfCubes, *edgeSizeOfCubes;

    int *leafParent;
    int *nodeParent;
    int *nodeI, *nodeCount;
    float *nodeValue;
    float *nodeX, *nodeY, *nodeZ;
    float *nodeVX, *nodeVY, *nodeVZ;

    __host__ __device__
    combineMergeTrees(unsigned long long *cubeMapping, unsigned long long *cubeId,
        long long *tmpNxt, long long *tmpFree,
        int *leafParent, int *nodeParent,
        int *nodeI, float *nodeValue, int *nodeCount,
        float *nodeX, float *nodeY, float *nodeZ,
        float *nodeVX, float *nodeVY, float *nodeVZ,
        unsigned long long *edgesSrc, unsigned long long *edgesDes, float *edgesWeight,
        int *edgeStartOfCubes, int *edgeSizeOfCubes,
        float min_ll, int sizeP, unsigned long long numOfCubesOri) :
        cubeMapping(cubeMapping), cubeId(cubeId),
        tmpNxt(tmpNxt), tmpFree(tmpFree),
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
      unsigned long long cubeStart = sizeP*i;
      unsigned long long cubeEnd   = (sizeP*(i+1)<numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

      unsigned long long cubeStartM = cubeMapping[cubeStart];
      unsigned long long cubeEndM   = cubeMapping[cubeEnd-1];

      // get the edges
      unsigned long long k=cubeStart;
      {
        for(int j=edgeStartOfCubes[k]; j<edgeStartOfCubes[k]+edgeSizeOfCubes[k]; j++)
        {
          int eSrc = edgesSrc[j];
          int eDes = edgesDes[j];

          float eWeight = edgesWeight[j];

          if(!(cubeId[eSrc]>=cubeStartM && cubeId[eSrc]<=cubeEndM) ||
             !(cubeId[eDes]>=cubeStartM && cubeId[eDes]<=cubeEndM))
          {
            edgeSizeOfCubes[k] -= (j-edgeStartOfCubes[k]);
            edgeStartOfCubes[k] = j;
            return;
          }


          // use this edge (e), to combine the merge trees
          //-----------------------------------------------------

          float weight = (eWeight < min_ll) ? min_ll : eWeight;

          // find the src & des nodes just below the required weight
          int src = leafParent[eSrc]; while(nodeParent[src]!=-1 && nodeValue[nodeParent[src]]<=weight) src = nodeParent[src];
          int des = leafParent[eDes]; while(nodeParent[des]!=-1 && nodeValue[nodeParent[des]]<=weight) des = nodeParent[des];

          // if src & des already have the same halo id, do NOT do anything
          if(nodeI[src]==nodeI[des]) continue;

          if(tmpNxt[cubeStart]==-1)
          {
            edgeSizeOfCubes[k] -= (j-edgeStartOfCubes[k]);
            edgeStartOfCubes[k] = j;
            return;
          }

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
        }
      }
    }
  };

  // jump node's parent pointers
  struct jumpNodePointers : public thrust::unary_function<int, void>
  {
    int   *nodeParent, *nodeCount;
    float *nodeValue;

    __host__ __device__
    jumpNodePointers(int *nodeParent, int *nodeCount, float *nodeValue) :
      nodeParent(nodeParent), nodeCount(nodeCount), nodeValue(nodeValue) {}

    __host__ __device__
    void operator()(int i)
    {
      int tmp = nodeParent[i];

      if(tmp!=-1 && nodeValue[i]==nodeValue[tmp])
        nodeCount[i] = -1; // set this node as free in FreeNodes function

      while(tmp!=-1 && nodeParent[tmp]!=-1 && nodeValue[tmp]==nodeValue[nodeParent[tmp]])
        tmp = nodeParent[tmp]; // continue

      nodeParent[i] = tmp; //jump pointers
    }
  };

  // jump leaf's parent pointers
  struct jumpLeafPointers : public thrust::unary_function<int, void>
  {
    int   *leafParent, *nodeParent;
    float *nodeValue;

    __host__ __device__
    jumpLeafPointers(int *leafParent, int *nodeParent, float *nodeValue) :
      leafParent(leafParent), nodeParent(nodeParent), nodeValue(nodeValue) {}

    __host__ __device__
    void operator()(int i)
    {
      int tmp = leafParent[i];

      while(tmp!=-1 && nodeParent[tmp]!=-1 && nodeValue[tmp]==nodeValue[nodeParent[tmp]])
        tmp = nodeParent[tmp]; // continue

      leafParent[i] = tmp; //jump pointers
    }
  };

  //check if a node is used
  struct isUsed
  {
    int *nodeCount;

    isUsed(int *nodeCount) : nodeCount(nodeCount) {}

    __host__ __device__
    bool operator()(const int i)
    {
      return (nodeCount[i]!=-1);
    }
  };

  // set current cube id for node
  struct setCubeId
  {
    long long *A, *B;

    int sizeP;

    setCubeId(long long *A, long long *B, int sizeP) : A(A), B(B), sizeP(sizeP) {}

    __host__ __device__
    void operator()(const int i)
    {
      B[i]  = (int)(A[i]/(2*sizeP));
      B[i] *= sizeP;
    }
  };

  // free nodes
  struct freeNodes : public thrust::unary_function<int, void>
  {
    long long *A, *D, *E;

    int  size1, size2;

    long long *tmpNxt, *tmpFree;

    int *nodeParent;
    int *nodeCount;

    __host__ __device__
    freeNodes(long long *A, long long *D, long long *E, long long *tmpNxt, long long *tmpFree,
        int *nodeParent, int *nodeCount, int size1, int size2) :
        A(A), D(D), E(E), tmpNxt(tmpNxt), tmpFree(tmpFree),
        nodeParent(nodeParent), nodeCount(nodeCount), size1(size1), size2(size2) {}

    __host__ __device__
    void operator()(int i)
    {
      int cubeStart = D[i];

      int size = (i==size2-1) ? size1-E[i] : E[i+1]-E[i];
      for(int k=E[i]; k<E[i]+size; k++)
      {
        int j = A[k];

        // set node as free
        if(nodeCount[j]==-1)
        {
          int tmpVal = tmpNxt[cubeStart];
          tmpNxt[cubeStart] = j;
          tmpFree[tmpNxt[cubeStart]] = tmpVal;

          nodeCount[j]  =  0;
          nodeParent[j] = -1;
        }
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

#ifndef HALO_MERGE_H
#define HALO_MERGE_H

#include <piston/halo.h>
#include <queue>

// When TEST is defined, output all results
//#define TEST

namespace piston
{

class halo_merge : public halo
{
public:
	float cubeLen;					// length of the cube
	double max_ll, min_ll;   // maximum & minimum linking lengths
	
	float totalTime; 				// total time taken for halo finding

	int mergetreeSize;      // total size of the global merge tree
  int numOfEdges;         // total number of edges in space

	int  side, size, ite; 	// variable needed to determine the neighborhood cubes

	unsigned int  numOfCubes;					 // total number of cubes in space
	int  cubesNonEmpty, cubesEmpty;		 // total number of nonempty & empty cubes in space
	int  cubesInX, cubesInY, cubesInZ; // number of cubes in each dimension

	int cubes, chunks; // amount of chunks & cube sizes used in computation steps, usually cubes is equal to cubesNonEmpty
	
  thrust::device_vector<int>   particleId; 						// for each particle, particle id
	thrust::device_vector<int>   particleSizeOfCubes; 	// number of particles in cubes
	thrust::device_vector<int>   particleStartOfCubes;	// stratInd of cubes  (within particleId)

  thrust::device_vector<int>   cubeId; // for each particle, cube id
  thrust::device_vector<int>   cubeMapping, cubeMappingInv; // mapping which seperates empty & nonempty cubes
  thrust::device_vector<int>   sizeOfChunks;  // size of each chunk of cubes
  thrust::device_vector<int>   startOfChunks; // start of each chunk of cubes

	thrust::device_vector<int>   edgesSrc, edgesDes; // edge of cubes - src & des
	thrust::device_vector<float> edgesWeight;	       // edge of cubes - weight
	thrust::device_vector<int>   edgeSizeOfCubes;    // size of edges in cubes
	thrust::device_vector<int>   edgeStartOfCubes;   // start of edges in cubes
	
	thrust::device_vector<int>   tmpIntArray;	// temperary arrays used
	thrust::device_vector<int>   tmpNxt, tmpFree;  // stores details of free items in merge tree

	halo_merge(float min_linkLength, float max_linkLength, std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1) : halo(filename, format, n, np, rL)
	{
		if(numOfParticles!=0)
		{
			struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3;

			//---- init stuff

		  // Unnormalize linkLengths so that it will work with box size distances
			min_ll  = min_linkLength*xscal; // get min_linkinglength
			max_ll  = max_linkLength*xscal; // get max_linkinglength
			cubeLen = min_ll / std::sqrt(3); // min_ll*min_ll = 3*cubeLen*cubeLen

			if(cubeLen <= 0) { std::cout << "--ERROR : please specify a valid cubeLen... current cubeLen is " << cubeLen/xscal << std::endl; return; }

			initDetails();

			std::cout << "lBoundS " << lBoundX << " " << lBoundY << " " << lBoundZ << std::endl;
			std::cout << "uBoundS " << uBoundX << " " << uBoundY << " " << uBoundZ << std::endl;

			//---- divide space into cubes
			gettimeofday(&begin, 0);
			divideIntoCubes();
			gettimeofday(&mid1, 0);

			std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ") ... cubeLen " << cubeLen/xscal << std::endl;

			//------- METHOD :
	    // parallel for each cube, create the local merge tree & get the set of edges.
			// globally combine the cubes, two cubes at a time by considering the edges

			gettimeofday(&mid2, 0);
			localStep();
			gettimeofday(&mid3, 0);

			std::cout << "-- localStep done" << std::endl;

			gettimeofday(&mid4, 0);
			globalStep();
			gettimeofday(&end, 0);

			std::cout << "-- globalStep done" << std::endl;

			checkValidMergeTree();
			getSizeOfMergeTree();
			clearSuperParents();
			writeMergeTreeToFile(filename);

			particleId.clear();	particleSizeOfCubes.clear(); particleStartOfCubes.clear();
			edgesSrc.clear();	 edgesDes.clear(); edgesWeight.clear(); edgeSizeOfCubes.clear(); edgeStartOfCubes.clear();
			cubeId.clear();	 cubeMapping.clear(); cubeMappingInv.clear();
			tmpIntArray.clear(); tmpNxt.clear(); tmpFree.clear(); sizeOfChunks.clear(); startOfChunks.clear();

			std::cout << std::endl;
			timersub(&mid1, &begin, &diff1);
			float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
			std::cout << "Time elapsed: " << seconds1 << " s for dividing space into cubes"<< std::endl << std::flush;
			timersub(&mid3, &mid2, &diff2);
			float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
			std::cout << "Time elapsed: " << seconds2 << " s for localStep - finding inter-cube edges"<< std::endl << std::flush;
			timersub(&end, &mid4, &diff3);
			float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
			std::cout << "Time elapsed: " << seconds3 << " s for globalStep - adjusting the merge trees"<< std::endl << std::flush;
			totalTime = seconds1 + seconds2 + seconds3;
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

		getHaloDetails();     // get the unique halo ids & set numOfHalos
		getHaloParticles();   // get the halo particles & set numOfHaloParticles
		setColors();          // set colors to halos
		writeHaloResults();	  // write halo results

		std::cout << "Number of Particles   : " << numOfParticles << std::endl;
		std::cout << "Number of Halos found : " << numOfHalos << std::endl;
		std::cout << "Merge tree size : " << mergetreeSize << " (leafs-" << numOfParticles << " other-" << mergetreeSize-numOfParticles << ")" << std::endl;
    std::cout << "Min_ll  : " << min_ll/xscal  << std::endl;
    std::cout << "Max_ll  : " << max_ll/xscal << std::endl << std::endl;
		std::cout << "-----------------------------" << std::endl << std::endl;
	}

	// find halo ids 
	void findHalos(float linkLength, int particleSize)
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setHaloId(thrust::raw_pointer_cast(&*leafValue.begin()),
				          thrust::raw_pointer_cast(&*leafCount.begin()),
				          thrust::raw_pointer_cast(&*leafI.begin()),
				          thrust::raw_pointer_cast(&*leafParent.begin()),
				          thrust::raw_pointer_cast(&*leafParentS.begin()),
						 		  thrust::raw_pointer_cast(&*haloIndex.begin()),
						     	linkLength, particleSize));
	}

	// for a given node set its halo id, for particles in filtered halos set id to -1
	struct setHaloId : public thrust::unary_function<int, void>
	{
		float *leafValue;
		int   *leafCount, *leafI, *leafParent, *leafParentS;
		int   *haloIndex;

		int    particleSize;
		float  linkLength;

		__host__ __device__
		setHaloId(float *leafValue, int *leafCount, int *leafI,
		  int *leafParent, int *leafParentS,
		  int *haloIndex, float linkLength, int particleSize) :
		  leafValue(leafValue), leafCount(leafCount), leafI(leafI),
		  leafParent(leafParent), leafParentS(leafParentS),
		  haloIndex(haloIndex), linkLength(linkLength), particleSize(particleSize) {}

		__host__ __device__
		void operator()(int i)
		{			
		  int n = i;

		  if(leafParentS[n]!=-1 && leafValue[leafParentS[n]]<=linkLength)
		    n = leafParentS[n];

		  while(leafParent[n]!=-1 && leafValue[leafParent[n]]<=linkLength)
        n = leafParent[n];

		  leafParentS[i] = n;

		  haloIndex[i] = (leafCount[n] >= particleSize) ? leafI[n] : -1;
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
				setHaloStats(thrust::raw_pointer_cast(&*leafCount.begin()),
                     thrust::raw_pointer_cast(&*leafX.begin()),
                     thrust::raw_pointer_cast(&*leafY.begin()),
                     thrust::raw_pointer_cast(&*leafZ.begin()),
                     thrust::raw_pointer_cast(&*leafVX.begin()),
                     thrust::raw_pointer_cast(&*leafVY.begin()),
                     thrust::raw_pointer_cast(&*leafVZ.begin()),
                     thrust::raw_pointer_cast(&*leafParentS.begin()),
										 thrust::raw_pointer_cast(&*tmpIntArray.begin()),
										 thrust::raw_pointer_cast(&*haloCount.begin()),
										 thrust::raw_pointer_cast(&*haloX.begin()),
										 thrust::raw_pointer_cast(&*haloY.begin()),
										 thrust::raw_pointer_cast(&*haloZ.begin()),
										 thrust::raw_pointer_cast(&*haloVX.begin()),
										 thrust::raw_pointer_cast(&*haloVY.begin()),
										 thrust::raw_pointer_cast(&*haloVZ.begin()),
										 linkLength, particleSize));
	}

	// for each halo, get its stats
	struct setHaloStats : public thrust::unary_function<int, void>
	{
	  int *leafCount, *leafParentS;
    float *leafX, *leafY, *leafZ;
    float *leafVX, *leafVY, *leafVZ;
		
		int    particleSize;
		float  linkLength;

		int *particleId;
		int *haloCount;
		float *haloX, *haloY, *haloZ;
		float *haloVX, *haloVY, *haloVZ;

		__host__ __device__
		setHaloStats(int *leafCount,
      float *leafX, float *leafY, float *leafZ,
      float *leafVX, float *leafVY, float *leafVZ,
      int *leafParentS, int *particleId, int *haloCount,
			float *haloX, float *haloY, float *haloZ,
			float *haloVX, float *haloVY, float *haloVZ,
			float linkLength, int particleSize) :
			leafCount(leafCount),
      leafX(leafX), leafY(leafY), leafZ(leafZ),
      leafVX(leafVX), leafVY(leafVY), leafVZ(leafVZ),
			leafParentS(leafParentS), particleId(particleId), haloCount(haloCount),
			haloX(haloX), haloY(haloY), haloZ(haloZ),
			haloVX(haloVX), haloVY(haloVY), haloVZ(haloVZ),
			linkLength(linkLength), particleSize(particleSize) {}

		__host__ __device__
		void operator()(int i)
		{			
		  int n = leafParentS[particleId[i]];

			haloCount[i] = leafCount[n];
			haloX[i] = (float)(leafX[n]/leafCount[n]);	haloVX[i] = (float)(leafVX[n]/leafCount[n]);
			haloY[i] = (float)(leafY[n]/leafCount[n]);	haloVY[i] = (float)(leafVY[n]/leafCount[n]);
			haloZ[i] = (float)(leafZ[n]/leafCount[n]);	haloVZ[i] = (float)(leafVZ[n]/leafCount[n]);
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
		
		thrust:: for_each(CountingIterator(0), CountingIterator(0)+numOfHaloParticles,
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

	// check whether the merge tree is valid or not
	void checkValidMergeTree()
	{
	  bool invalid = false;
	  for(int i=0; i<numOfParticles; i++)
	  {
	    int j = i;

	    int count = 0;
	    while(j!=-1 && leafValue[j]<=min_ll) { j = leafParent[j]; count++; }

      // if a node has more than one parent node with its value <= min_ll, the tree is invalid
      if(count > 2) { invalid = true;   std::cout << i << " " << count << std::endl; break; }
	  }
    std::cout << std::endl;

    if(invalid) std::cout << "-- ERROR: invalid merge tree " << std::endl;
    else std::cout << "-- valid merge tree " << std::endl;
	}

	// get the size of the merge tree
	void getSizeOfMergeTree()
	{
	  mergetreeSize = numOfParticles + (numOfParticles - thrust::count(leafI.begin()+numOfParticles, leafI.end(), -1));
	}

	// clear super parents of all node
	void clearSuperParents()
	{
		thrust::fill(leafParentS.begin(), leafParentS.end(), -1);
	}

	//write merge tree to a file
	void writeMergeTreeToFile(std::string filename)
	{
	  float fBlock[4];

	  // write particle details - .particle file (pId x y z)
	  {
      std::ofstream *outStream1 = new std::ofstream();
      outStream1->open((filename+".particle").c_str()); //std::ios::out|std::ios::binary
      for(int i=0; i<numOfParticles; i++)
      {
        fBlock[0] = i;   fBlock[1] = leafX[i];   fBlock[2] = leafY[i];   fBlock[3] = leafZ[i];

        (*outStream1) << i << " ";
        (*outStream1) << leafX[i] << " ";
        (*outStream1) << leafY[i] << " ";
        (*outStream1) << leafZ[i] << " ";
        (*outStream1) << "\n";

        //outStream1->write(reinterpret_cast<const char*>(fBlock), 4 * sizeof(float));
      }
      outStream1->close();
	  }

	  int start = 0;
    for(int i=0; i<numOfParticles; i++)
      if(leafParent[i]==-1 || leafValue[leafParent[i]]>min_ll) start++;

    tmpIntArray.resize(numOfParticles);
    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
        checkIfNode(thrust::raw_pointer_cast(&*leafI.begin()),
                    thrust::raw_pointer_cast(&*tmpIntArray.begin()),
                    numOfParticles));
    thrust::exclusive_scan(tmpIntArray.begin(), tmpIntArray.end(), tmpIntArray.begin(), start);

	  // write feature details - .feature file (fId birth death parent) & .segmentation file (particlesIds)
	  {
      std::ofstream *outStream2 = new std::ofstream();
      outStream2->open((filename+".feature").c_str());

      std::ofstream *outStream3 = new std::ofstream();
      outStream3->open((filename+".segmentation").c_str());

      int offset = 0;

	    //first write the features with just one particle
      for(int i=0; i<numOfParticles; i++)
      {
        int size = 0;
        if(leafParent[i]==-1)
        {
          size++;
          (*outStream2) << offset << " ";
          (*outStream2) << min_ll/xscal << " ";
          (*outStream2) << max_ll/xscal << " ";
          (*outStream2) << -1 << " ";
          (*outStream2) << offset << " " << size << "\n";

          (*outStream3) << offset << " ";
        }
        else if(leafValue[leafParent[i]]>min_ll)
        {
          size++;
          (*outStream2) << offset << " ";
          (*outStream2) << min_ll/xscal << " ";
          (*outStream2) << leafValue[leafParent[i]]/xscal << " ";
          (*outStream2) << tmpIntArray[leafParent[i]-numOfParticles] << " ";
          (*outStream2) << offset << " " << size << "\n";

          (*outStream3) << offset << " ";
        }
        offset += size;
      }

      //write out rest of the parent nodes in the tree
      for(int i=0; i<numOfParticles; i++)
      {
        if(leafI[numOfParticles+i]!=-1)
        {
          (*outStream2) << tmpIntArray[i] << " ";
          (*outStream2) << leafValue[numOfParticles+i]/xscal << " ";
          (*outStream2) << ((leafParent[numOfParticles+i]!=-1) ? leafValue[leafParent[numOfParticles+i]]/xscal : max_ll/xscal) << " ";
          (*outStream2) << ((leafParent[numOfParticles+i]!=-1) ? tmpIntArray[leafParent[numOfParticles+i]-numOfParticles] : -1) << " ";

          int size = 0;
          //get this features particles
          std::queue<int> children;
          children.push(numOfParticles+i);
          while(!children.empty())
          {
            int j = children.front();
            children.pop();

            if(leafValue[j]==0) { (*outStream3) << j << " ";  size++; }

            int child = leafChildS[j];
            while(child!=-1)
            {
              children.push(child);
              child = leafSibling[child];
            }
          }

          (*outStream2) << offset << " " << size << "\n";
          offset += size;
        }
      }
      outStream2->close();
      outStream3->close();
	  }
	}
	
	// for a given node set its halo id, for particles in filtered halos set id to -1
  struct checkIfNode : public thrust::unary_function<int, void>
  {
    int *leafI, *tmp;
    int numOfParticles;

    __host__ __device__
    checkIfNode(int *leafI, int *tmp, int numOfParticles) :
      leafI(leafI), tmp(tmp), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {
      tmp[i] = (leafI[numOfParticles+i]==-1) ? 0 : 1;
    }
  };



	//------- init stuff
	void initDetails()
	{
		initParticleIds();	// set particle ids
		initNodeDetails();  // set node details
		setNumberOfCubes();	// get total number of cubes
	}

  // set initial particle ids
	void initParticleIds()
	{
		particleId.resize(numOfParticles);
		thrust::sequence(particleId.begin(), particleId.end());
	}

	//set initial node details
	void initNodeDetails()
	{
	  // resize node details
    leafX.resize(2*numOfParticles);
    leafY.resize(2*numOfParticles);
    leafZ.resize(2*numOfParticles);
    leafVX.resize(2*numOfParticles);
    leafVY.resize(2*numOfParticles);
    leafVZ.resize(2*numOfParticles);
    leafM.resize(2*numOfParticles);
    leafI.resize(2*numOfParticles);
    leafCount.resize(2*numOfParticles);
    leafValue.resize(2*numOfParticles);
    leafParent.resize(2*numOfParticles);
    leafParentS.resize(2*numOfParticles);
    leafChildS.resize(2*numOfParticles);
    leafChildE.resize(2*numOfParticles);
    leafSibling.resize(2*numOfParticles);

    thrust::fill(leafCount.begin(),   leafCount.begin()+numOfParticles, 1);
    thrust::fill(leafValue.begin(),   leafValue.begin()+numOfParticles,  0.0);
    thrust::fill(leafParent.begin(),  leafParent.end(),  -1);
    thrust::fill(leafParentS.begin(), leafParentS.end(), -1);
    thrust::fill(leafChildS.begin(),  leafChildS.end(),  -1);
    thrust::fill(leafChildE.begin(),  leafChildE.end(),  -1);
    thrust::fill(leafSibling.begin(), leafSibling.end(), -1);
    thrust::fill(leafI.begin()+numOfParticles, leafI.end(), -1);
	}

	// get total number of cubes
	void setNumberOfCubes()
	{
		cubesInX = (std::ceil((uBoundX-lBoundX)/cubeLen) == 0) ? 1 : std::ceil((uBoundX-lBoundX)/cubeLen);
		cubesInY = (std::ceil((uBoundY-lBoundY)/cubeLen) == 0) ? 1 : std::ceil((uBoundY-lBoundY)/cubeLen);
		cubesInZ = (std::ceil((uBoundZ-lBoundZ)/cubeLen) == 0) ? 1 : std::ceil((uBoundZ-lBoundZ)/cubeLen);

		numOfCubes = cubesInX*cubesInY*cubesInZ; // set number of cubes
	}



	//------- divide space into cubes
	void divideIntoCubes()
	{
	  struct timeval begin, mid1, mid2, end, diff0, diff1, diff2;
	  gettimeofday(&begin, 0);
		setCubeIds();		      		// for each particle, set cube id
	  gettimeofday(&mid1, 0);
		sortParticlesByCubeID();  // sort Particles by cube Id
	  gettimeofday(&mid2, 0);
		getSizeAndStartOfCubes(); // for each cube, count its particles
	  gettimeofday(&end, 0);

    #ifdef TEST
      outputCubeDetails("init cube details"); // output cube details
    #endif

    std::cout << std::endl;
	  std::cout << "'divideIntoCubes' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff0);
    float seconds0 = diff0.tv_sec + 1.0E-6*diff0.tv_usec;
    std::cout << "Time elapsed0: " << seconds0 << " s for setCubeIds"<< std::endl << std::flush;
    timersub(&mid2, &mid1, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << " s for sortParticlesByCubeID"<< std::endl << std::flush;
    timersub(&end, &mid2, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed2: " << seconds2 << " s for getSizeAndStartOfCubes"<< std::endl << std::flush;
	}

	// set cube ids of particles
	void setCubeIds()
	{
	  cubeId.resize(numOfParticles);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setCubeIdOfParticle(thrust::raw_pointer_cast(&*leafX.begin()),
                            thrust::raw_pointer_cast(&*leafY.begin()),
                            thrust::raw_pointer_cast(&*leafZ.begin()),
														thrust::raw_pointer_cast(&*cubeId.begin()),
														cubeLen, lBoundX, lBoundY, lBoundZ, cubesInX, cubesInY, cubesInZ));
	}

	// for a given particle, set its cube id
	struct setCubeIdOfParticle : public thrust::unary_function<int, void>
	{
		float  cubeLen;
		float  lBoundX, lBoundY, lBoundZ;
		int    cubesInX, cubesInY, cubesInZ;

		int   *cubeId;
		float *leafX, *leafY, *leafZ;

		__host__ __device__
		setCubeIdOfParticle(float *leafX, float *leafY, float *leafZ,
		  int *cubeId, float cubeLen,
		  float lBoundX, float lBoundY, float lBoundZ,
			int cubesInX, int cubesInY, int cubesInZ) :
			leafX(leafX), leafY(leafY), leafZ(leafZ),
			cubeId(cubeId), cubeLen(cubeLen),
			lBoundX(lBoundX), lBoundY(lBoundY), lBoundZ(lBoundZ),
			cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ){}

		__host__ __device__
		void operator()(int i)
		{
			int n = i;
	
			// get x,y,z coordinates for the cube
			int z = (((leafZ[n]-lBoundZ)/cubeLen)>=cubesInZ) ? cubesInZ-1 : (leafZ[n]-lBoundZ)/cubeLen;
			int y = (((leafY[n]-lBoundY)/cubeLen)>=cubesInY) ? cubesInY-1 : (leafY[n]-lBoundY)/cubeLen;
			int x = (((leafX[n]-lBoundX)/cubeLen)>=cubesInX) ? cubesInX-1 : (leafX[n]-lBoundX)/cubeLen;
			
			cubeId[i] = (z*(cubesInX*cubesInY) + y*cubesInX + x); // get cube id
		}
	};

	// sort particles by cube id
	void sortParticlesByCubeID()
	{
    thrust::stable_sort_by_key(cubeId.begin(), cubeId.end(), particleId.begin());
	}
		
	// sort cube id by particles
	void sortCubeIDByParticles()
	{
    thrust::stable_sort_by_key(particleId.begin(), particleId.end(), cubeId.begin());
	}

//------------ TODO : implement in a better way
	// for each cube, get the size & start of cube particles (in particleId array)
	void getSizeAndStartOfCubes()
	{
		int num = (numOfParticles<numOfCubes) ? numOfParticles : numOfCubes;

		cubeMapping.resize(num);
		particleSizeOfCubes.resize(num);

		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
		new_end = thrust::reduce_by_key(cubeId.begin(), cubeId.end(), ConstantIterator(1), cubeMapping.begin(), particleSizeOfCubes.begin());

		cubesNonEmpty = thrust::get<0>(new_end) - cubeMapping.begin();
		cubesEmpty    = numOfCubes - cubesNonEmpty;	

		cubes = cubesNonEmpty; // get the cubes which should be considered

		// get the mapping for nonempty cubes
		cubeMappingInv.resize(numOfCubes);
		thrust::fill(cubeMappingInv.begin(), cubeMappingInv.end(), -1);
		thrust::scatter(CountingIterator(0), CountingIterator(0)+cubesNonEmpty, cubeMapping.begin(), cubeMappingInv.begin());

		// get the size & start details for only non empty cubes
		particleStartOfCubes.resize(cubesNonEmpty);
		thrust::exclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubesNonEmpty, particleStartOfCubes.begin());
	}
	


	//------- output results

	// print cube details from device vectors
	void outputCubeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << "-- Dim    (" << lBoundX << "," << lBoundY << "," << lBoundZ << "), (";
		std::cout << uBoundX << "," << uBoundY << "," << uBoundZ << ")" << std::endl;
		std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;
		std::cout << std::endl << "----------------------" << std::endl << std::endl;

		std::cout << "particleId 	"; thrust::copy(particleId.begin(), particleId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "cubeID		"; thrust::copy(cubeId.begin(), cubeId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << std::endl;
		std::cout << "cubeMapping  "; thrust::copy(cubeMapping.begin(), cubeMapping.begin()+cubesNonEmpty, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubesNonEmpty, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "startOfCube	"; thrust::copy(particleStartOfCubes.begin(), particleStartOfCubes.begin()+cubesNonEmpty, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "----------------------" << std::endl << std::endl;
	}

	// print edge details from device vectors
	void outputEdgeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << "numOfEdges			 " << numOfEdges << std::endl << std::endl;
		std::cout << "edgeSizeOfCubes	 "; thrust::copy(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubesNonEmpty, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "edgeStartOfCubes "; thrust::copy(edgeStartOfCubes.begin(), edgeStartOfCubes.begin()+cubesNonEmpty, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << std::endl;

		for(int i=0; i<cubesNonEmpty; i++)
		{
			for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)	
				std::cout << "---- " << edgesSrc[j] << "," << edgesDes[j] << "," << edgesWeight[j] <<  ")" << std::endl;
		}
		std::cout << std::endl << "----------------------" << std::endl << std::endl;
	}
	


	//------- METHOD : parallel for each cube, get the set of edges & create the submerge tree. Globally combine them

	//---------------- METHOD - Local functions

	// locally, get intra-cube edges for each cube & create the local merge trees
	void localStep()
	{
		// resize vectors necessary for merge tree construction
		tmpNxt.resize(cubes);
		tmpFree.resize(numOfParticles);

    struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff1, diff2, diff3, diff4, diff5, diff6;
    gettimeofday(&begin, 0);
    thrust::sequence(tmpFree.begin(), tmpFree.end(), 1);
    gettimeofday(&mid1, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				initNxt(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
								thrust::raw_pointer_cast(&*tmpNxt.begin()),
								thrust::raw_pointer_cast(&*tmpFree.begin())));
    gettimeofday(&mid2, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				createSubMergeTree(thrust::raw_pointer_cast(&*leafParent.begin()),
                           thrust::raw_pointer_cast(&*leafChildS.begin()),
                           thrust::raw_pointer_cast(&*leafChildE.begin()),
                           thrust::raw_pointer_cast(&*leafSibling.begin()),
                           thrust::raw_pointer_cast(&*leafI.begin()),
                           thrust::raw_pointer_cast(&*leafValue.begin()),
                           thrust::raw_pointer_cast(&*leafCount.begin()),
				                   thrust::raw_pointer_cast(&*leafX.begin()),
                           thrust::raw_pointer_cast(&*leafY.begin()),
                           thrust::raw_pointer_cast(&*leafZ.begin()),
                           thrust::raw_pointer_cast(&*leafVX.begin()),
                           thrust::raw_pointer_cast(&*leafVY.begin()),
                           thrust::raw_pointer_cast(&*leafVZ.begin()),
													 thrust::raw_pointer_cast(&*particleId.begin()),
													 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
												 	 thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
													 thrust::raw_pointer_cast(&*tmpNxt.begin()),
													 thrust::raw_pointer_cast(&*tmpFree.begin()),
													 min_ll, numOfParticles));
		gettimeofday(&mid3, 0);
		initArrays();  				    // init arrays needed for storing edges
		gettimeofday(&mid4, 0);
		getEdgesPerCube(); 				// for each cube, get the set of edges
		gettimeofday(&mid5, 0);
		sortCubeIDByParticles();	// sort cube ids by particle id
    gettimeofday(&end, 0);

		#ifdef TEST
			outputMergeTreeDetails("The local merge trees.."); // output merge tree details
			outputEdgeDetails("Edges to be considered in the global step.."); // output edge details
		#endif

    std::cout << std::endl;
	  std::cout << "'localStep' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed0: " << seconds1 << " s for initNodes " << std::endl << std::flush;
		timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed1: " << seconds2 << " s for initNxt " << std::endl << std::flush;
		timersub(&mid3, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed2: " << seconds3 << " s for createSubMergeTree " << std::endl << std::flush;   
		timersub(&mid4, &mid3, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed3: " << seconds4 << " s for initEdgeArrays " << std::endl << std::flush;
		timersub(&mid5, &mid4, &diff5);
    float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
    std::cout << "Time elapsed4: " << seconds5 << " s for getEdgesPerCube " << std::endl << std::flush;
		timersub(&end, &mid5, &diff6);
    float seconds6 = diff6.tv_sec + 1.0E-6*diff6.tv_usec;
    std::cout << "Time elapsed5: " << seconds6 << " s for sortCubeIDByParticles " << std::endl << std::flush;
	}

  // finalize the init of tmpFree & tmpNxt arrays
  struct initNxt : public thrust::unary_function<int, void>
  {
    int  *particleSizeOfCubes, *particleStartOfCubes;
		int  *tmpFree, *tmpNxt;

		__host__ __device__
		initNxt(int *particleSizeOfCubes, int *particleStartOfCubes, 
			int *tmpNxt, int *tmpFree) :
			particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes),			
			tmpNxt(tmpNxt), tmpFree(tmpFree) {}

    __host__ __device__
    void operator()(int i)
    {
			int start = particleStartOfCubes[i];
			int size  = particleSizeOfCubes[i];

		  tmpNxt[i] = (size>0) ? start : -1;
			tmpFree[start+size-1] = -1;
    }
 	};

	// create the submerge tree for each cube
	struct createSubMergeTree : public thrust::unary_function<int, void>
  {
	  int *leafParent, *leafChildS, *leafChildE, *leafSibling;
	  int *leafI, *leafCount;

	  float *leafValue;
	  float *leafX, *leafY, *leafZ;
	  float *leafVX, *leafVY, *leafVZ;

		int *particleSizeOfCubes, *particleStartOfCubes, *particleId;
		int *tmpNxt, *tmpFree;

	  float min_ll;
	  int numOfParticles;

		__host__ __device__
		createSubMergeTree(int *leafParent, int *leafChildS, int *leafChildE, int *leafSibling,
		  int *leafI, float *leafValue, int *leafCount,
		  float *leafX, float *leafY, float *leafZ,
		  float *leafVX, float *leafVY, float *leafVZ,
		  int *particleId, int *particleSizeOfCubes, int *particleStartOfCubes,
			int *tmpNxt, int *tmpFree, float min_ll, int numOfParticles) :
			leafParent(leafParent), leafChildS(leafChildS), leafChildE(leafChildE), leafSibling(leafSibling),
			leafI(leafI), leafValue(leafValue), leafCount(leafCount),
			leafX(leafX), leafY(leafY), leafZ(leafZ),
			leafVX(leafVX), leafVY(leafVY), leafVZ(leafVZ),
			particleId(particleId), particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes),
			tmpNxt(tmpNxt), tmpFree(tmpFree), min_ll(min_ll), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {
			if(particleSizeOfCubes[i]<=1) return;
			
			// get the next free node & set it as the parent
			int n = tmpNxt[i] + numOfParticles;
			int tmpVal = tmpFree[tmpNxt[i]];
			tmpFree[tmpNxt[i]] = -2;
			tmpNxt[i] = tmpVal;

			float x=0,  y=0,  z=0;
			float vx=0, vy=0, vz=0;

			int minHaloId = -1;
			for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
			{
				int tmp = particleId[j];

				leafParent[tmp] = n;
				
				if(leafChildS[n]==-1) { leafChildS[n]=tmp; leafChildE[n]=tmp; }
				else     { leafSibling[leafChildE[n]]=tmp; leafChildE[n]=tmp; }

				minHaloId = (minHaloId==-1) ? leafI[tmp] : (minHaloId<leafI[tmp] ? minHaloId : leafI[tmp]);

				x+=leafX[tmp]; vx+=leafVX[tmp];
				y+=leafY[tmp]; vy+=leafVY[tmp];
				z+=leafZ[tmp]; vz+=leafVZ[tmp];
			}

			leafI[n] = minHaloId;
			leafValue[n] = min_ll;
      leafCount[n] = particleSizeOfCubes[i];
      leafX[n] = x;   leafVX[n] = vx;
      leafY[n] = y;   leafVY[n] = vy;
      leafZ[n] = z;   leafVZ[n] = vz;
    }
 	};

  // for each cube, init arrays needed for storing edges
	void initArrays()
	{
		// for each vube, get the details of how many neighbors should be checked 
		side = (1 + std::ceil(max_ll/cubeLen)*2);
		size = side*side*side;
		ite = (size-1)/2;

		std::cout << std::endl << "side " << side << " cubeSize " << size << " ite " << ite << std::endl << std::endl;
		std::cout << cubesEmpty << " of " << numOfCubes << " cubes are empty. (" << (((double)cubesEmpty*100)/(double)numOfCubes) << "%) ... non empty cubes " << cubesNonEmpty << std::endl;

		edgeSizeOfCubes.resize(cubes);
		edgeStartOfCubes.resize(cubes);

		// for each cube, get neighbor details
		tmpIntArray.resize(cubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				getNeighborDetails(thrust::raw_pointer_cast(&*cubeMapping.begin()),
													 thrust::raw_pointer_cast(&*cubeMappingInv.begin()),
													 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
													 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
												   thrust::raw_pointer_cast(&*tmpIntArray.begin()),
												   ite, side, cubesInX, cubesInY, cubesInZ));
		setChunks(); // group cubes in to chunks
		tmpIntArray.clear();
	
		// for each cube, set the space required for storing edges		
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes, edgeStartOfCubes.begin());
		numOfEdges = edgeStartOfCubes[cubes-1] + edgeSizeOfCubes[cubes-1]; // size of edges array 

		std::cout << std::endl << "numOfEdges before " << numOfEdges << std::endl;

		// init edge arrays
		edgesSrc.resize(numOfEdges);
		edgesDes.resize(numOfEdges);
		edgesWeight.resize(numOfEdges);
	}

	//for each cube, sum the number of particles in neighbor cubes & get the sum of non empty neighbor cubes
	struct getNeighborDetails : public thrust::unary_function<int, void>
  {
		int *cubeMapping, *cubeMappingInv;
		int *particleSizeOfCubes;
		int *tmpIntArray, *edgeSizeOfCubes;

		int  ite, side;
		int  cubesInX, cubesInY, cubesInZ;

		__host__ __device__
		getNeighborDetails(int *cubeMapping, int *cubeMappingInv, 
				int *particleSizeOfCubes, int *edgeSizeOfCubes, int *tmpIntArray, 
				int ite, int side, int cubesInX, int cubesInY, int cubesInZ) : 
				cubeMapping(cubeMapping), cubeMappingInv(cubeMappingInv), 
				particleSizeOfCubes(particleSizeOfCubes), edgeSizeOfCubes(edgeSizeOfCubes), tmpIntArray(tmpIntArray), 
				ite(ite), side(side), cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ) {}

    __host__ __device__
    void operator()(int i)
    {
			int i_mapped = cubeMapping[i];

			// get x,y,z coordinates for the cube
			int tmp = i_mapped % (cubesInX*cubesInY);
			int z = i_mapped / (cubesInX*cubesInY);
			int y = tmp / cubesInX;
			int x = tmp % cubesInX;

			int len = (side-1)/2;

			for(int num=0; num<ite; num++)
			{
				int tmp1 = num % (side*side);
				int z1 = num / (side*side);
				int y1 = tmp1 / side;
				int x1 = tmp1 % side;		

				// get x,y,z coordinates for the current cube 
				int currentX = x - len + x1;
				int currentY = y - len + y1;
				int currentZ = z - len + z1;

				int cube_mapped = -1, cube = -1;
				if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
				{
					cube_mapped = (currentZ*(cubesInY*cubesInX) + currentY*cubesInX + currentX);
					cube = cubeMappingInv[cube_mapped];
				}

				if(cube_mapped==-1 || cube==-1 || particleSizeOfCubes[i]==0 || particleSizeOfCubes[cube]==0) continue;

				edgeSizeOfCubes[i]++;  //sum the non empty neighbor cubes
				tmpIntArray[i] += particleSizeOfCubes[cube]; // sum the number of particles in neighbor cubes
			}

			tmpIntArray[i] *= particleSizeOfCubes[i]; // multiply by particles in this cube
    }
 	};

//------------ TODO : Do accurate chunking, Look at whether I am doing the right thing in chunking
	// group the set of cubes to chunks of same computation sizes (for load balance)
	void setChunks()
	{
		thrust::device_vector<int>::iterator maxSize = thrust::max_element(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes);

		startOfChunks.resize(cubes);
		sizeOfChunks.resize(cubes);

		thrust::fill(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
		thrust::inclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes, sizeOfChunks.begin());
/*
		std::cout << "particleSizeOfCubes	 "; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "inclusive_scan	 "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
*/
		thrust::copy_if(CountingIterator(0), CountingIterator(0)+cubes, startOfChunks.begin(),
				isStartOfChunks(thrust::raw_pointer_cast(&*sizeOfChunks.begin()), *maxSize));	
		chunks = cubes - thrust::count(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
				setSizeOfChunks(thrust::raw_pointer_cast(&*startOfChunks.begin()),
												thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
												chunks, cubes));

		std::cout << "maxSize " << *maxSize << " chunks " << chunks << std::endl;
/*
		std::cout << "startOfChunks	 "; thrust::copy(startOfChunks.begin(), startOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "sizeOfChunks	 "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;

		std::cout << "amountOfChunks	 ";
		for(int i=0; i<chunks; i++)
		{
			int count = 0;
			for(int j=startOfChunks[i]; j<startOfChunks[i]+sizeOfChunks[i]; j++)
				count += particleSizeOfCubes[j];
			std::cout << count << " ";
		}
		std::cout << std::endl << std::endl;
*/
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
	
			int a = sizeOfChunks[i] / max;	int b = sizeOfChunks[i-1] / max;	
			int d = sizeOfChunks[i] % max;	int e = sizeOfChunks[i-1] % max;

			if((a!=b && d==0 && e==0) ||(a!=b && d!=0) || (a==b && e==0)) return true;

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
			else 						sizeOfChunks[i] = startOfChunks[i+1] - startOfChunks[i];
    }
 	};

	// for each cube, get the set of edges by running them in chunks of cubes
	void getEdgesPerCube()
	{	
		thrust::fill(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes,0);		
		
		thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
				getEdges(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
								 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*startOfChunks.begin()),
								 thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
								 thrust::raw_pointer_cast(&*cubeMapping.begin()),
								 thrust::raw_pointer_cast(&*cubeMappingInv.begin()),
                 thrust::raw_pointer_cast(&*leafX.begin()),
                 thrust::raw_pointer_cast(&*leafY.begin()),
                 thrust::raw_pointer_cast(&*leafZ.begin()),
								 thrust::raw_pointer_cast(&*particleId.begin()),
								 max_ll, min_ll, ite, cubesInX, cubesInY, cubesInZ, side,
								 thrust::raw_pointer_cast(&*edgesSrc.begin()),
	               thrust::raw_pointer_cast(&*edgesDes.begin()),
	               thrust::raw_pointer_cast(&*edgesWeight.begin()),
								 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));		

		numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes); // set the correct number of edges

		std::cout << "numOfEdges after " << numOfEdges << std::endl;
	}

//------------ TODO : try to do the edge calculation parallel for the number of iterations as well
	// for each cube, get the set of edges after comparing
	struct getEdges : public thrust::unary_function<int, void>
	{
		int    ite;
		float  max_ll, min_ll;
		float *leafX, *leafY, *leafZ;

		int   *startOfChunks, *sizeOfChunks;
		int   *cubeMapping, *cubeMappingInv;
		int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;

		int    side;
		int    cubesInX, cubesInY, cubesInZ;
		
		int   *edgesSrc, *edgesDes;
		float *edgesWeight;
		int   *edgeStartOfCubes, *edgeSizeOfCubes;

		__host__ __device__
		getEdges(int *particleStartOfCubes, int *particleSizeOfCubes, 
				int *startOfChunks, int *sizeOfChunks,
				int *cubeMapping, int *cubeMappingInv,
				float *leafX, float *leafY, float *leafZ,
				int *particleId, float max_ll, float min_ll, int ite,
				int cubesInX, int cubesInY, int cubesInZ, int side, 
				int *edgesSrc, int *edgesDes, float *edgesWeight,
				int *edgeSizeOfCubes, int *edgeStartOfCubes) :
 				particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
				startOfChunks(startOfChunks), sizeOfChunks(sizeOfChunks),
				cubeMapping(cubeMapping), cubeMappingInv(cubeMappingInv),
				leafX(leafX), leafY(leafY), leafZ(leafZ),
				particleId(particleId), max_ll(max_ll), min_ll(min_ll), ite(ite), 
				cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), side(side),
				edgesSrc(edgesSrc), edgesDes(edgesDes), edgesWeight(edgesWeight),
				edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

		__host__ __device__
		void operator()(int i)
		{	
			for(int l=startOfChunks[i]; l<startOfChunks[i]+sizeOfChunks[i]; l++)
			{		
				int i_mapped = cubeMapping[l];

				// get x,y,z coordinates for the cube
				int tmp = i_mapped % (cubesInX*cubesInY);
				int z = i_mapped / (cubesInX*cubesInY);
				int y = tmp / cubesInX;
				int x = tmp % cubesInX;

				int len = (side-1)/2;

				for(int num=0; num<ite; num++)
				{
					int tmp1 = num % (side*side);
					int z1 = num / (side*side);
					int y1 = tmp1 / side;
					int x1 = tmp1 % side;		

					// get x,y,z coordinates for the current cube 
					int currentX = x - len + x1;
					int currentY = y - len + y1;
					int currentZ = z - len + z1;

					int cube_mapped = -1, cube = -1;
					if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
					{
						cube_mapped = (currentZ*(cubesInY*cubesInX) + currentY*cubesInX + currentX);
						cube = cubeMappingInv[cube_mapped];
					}

					if(cube_mapped==-1 || cube==-1 || particleSizeOfCubes[l]==0 || particleSizeOfCubes[cube]==0) continue;

					int eSrc, eDes;
					float eWeight;

					// for each particle in this cube
					float dist_min = max_ll+1;
					for(int j=particleStartOfCubes[l]; j<particleStartOfCubes[l]+particleSizeOfCubes[l]; j++)
					{
						int pId_j = particleId[j];

						// compare with particles in neighboring cube
						for(int k=particleStartOfCubes[cube]; k<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; k++)
						{
							int pId_k = particleId[k];

							double xd = (leafX[pId_j]-leafX[pId_k]);  if (xd < 0.0f) xd = -xd;
							double yd = (leafY[pId_j]-leafY[pId_k]);  if (yd < 0.0f) yd = -yd;
							double zd = (leafZ[pId_j]-leafZ[pId_k]);  if (zd < 0.0f) zd = -zd;

							if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
							{
								double dist = (double)std::sqrt(xd*xd + yd*yd + zd*zd);

								if(dist <= max_ll && dist < dist_min)
								{
									int srcV = (pId_j <= pId_k) ? pId_j : pId_k;
									int desV = (srcV == pId_k)  ? pId_j : pId_k;

									dist_min = dist;

									eSrc = srcV;
									eDes = desV;
									eWeight = dist;

									if(dist_min <= min_ll) goto loop;
								}
							}
						}
					}

					// add edge
					loop:
					if(dist_min < max_ll + 1)
					{
					  edgesSrc[edgeStartOfCubes[l] + edgeSizeOfCubes[l]] = eSrc;
            edgesDes[edgeStartOfCubes[l] + edgeSizeOfCubes[l]] = eDes;
            edgesWeight[edgeStartOfCubes[l] + edgeSizeOfCubes[l]] = eWeight;
						edgeSizeOfCubes[l]++;
					}
				}
			}
		}
	};




	//---------------- METHOD - Global functions

  // combine two local merge trees, two cubes at a time
	void globalStep()
	{
		int cubesOri = cubes;
		int sizeP = 2;

		// set new number of cubes
		int cubesOld = cubes;
		cubes = (int)std::ceil(((double)cubes/2));

		std::cout << std::endl;

		if(numOfEdges==0) return;

		// iteratively combine the cubes two at a time
		while(cubes!=cubesOld && cubes>0)
		{
	    struct timeval begin, end, diff;
	    gettimeofday(&begin, 0);
			thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes, 
				combineFreeLists(thrust::raw_pointer_cast(&*tmpNxt.begin()),
												 thrust::raw_pointer_cast(&*tmpFree.begin()),
												 sizeP, cubesOri));
			thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				combineMergeTrees(thrust::raw_pointer_cast(&*cubeMapping.begin()),
													thrust::raw_pointer_cast(&*cubeId.begin()),
													thrust::raw_pointer_cast(&*tmpNxt.begin()),
													thrust::raw_pointer_cast(&*tmpFree.begin()),
													thrust::raw_pointer_cast(&*leafParent.begin()),
                          thrust::raw_pointer_cast(&*leafChildS.begin()),
                          thrust::raw_pointer_cast(&*leafChildE.begin()),
                          thrust::raw_pointer_cast(&*leafSibling.begin()),
                          thrust::raw_pointer_cast(&*leafI.begin()),
                          thrust::raw_pointer_cast(&*leafValue.begin()),
                          thrust::raw_pointer_cast(&*leafCount.begin()),
                          thrust::raw_pointer_cast(&*leafX.begin()),
                          thrust::raw_pointer_cast(&*leafY.begin()),
                          thrust::raw_pointer_cast(&*leafZ.begin()),
                          thrust::raw_pointer_cast(&*leafVX.begin()),
                          thrust::raw_pointer_cast(&*leafVY.begin()),
                          thrust::raw_pointer_cast(&*leafVZ.begin()),
                          thrust::raw_pointer_cast(&*edgesSrc.begin()),
                          thrust::raw_pointer_cast(&*edgesDes.begin()),
                          thrust::raw_pointer_cast(&*edgesWeight.begin()),
													thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
													thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
													min_ll, sizeP, cubesOri, numOfParticles));
			gettimeofday(&end, 0);

      timersub(&end, &begin, &diff);
      float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
      std::cout << "Time elapsed: " << seconds << " s for nonEmptyCubes " << cubes;

			numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubesOri);
			std::cout << " numOfEdges after " << numOfEdges << std::endl;

			// set new number of cubes & sizeP
			sizeP *= 2;
			cubesOld = cubes;
			cubes = (int)std::ceil(((double)cubes/2));
		}
	}

	// combine free nodes lists of each cube at this iteration
	struct combineFreeLists : public thrust::unary_function<int, void>
  {
		int   sizeP, numOfCubesOri;

		int  *tmpNxt, *tmpFree;

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
				while(nxt!=-1 && tmpFree[nxt]!=-1)  nxt = tmpFree[nxt];

				k += sizeP/2;
				if(k<cubeEnd) tmpFree[nxt] = tmpNxt[k];
			}
		}
	};

	// combine two local merge trees
	struct combineMergeTrees : public thrust::unary_function<int, void>
  {
    float  min_ll;
		int    sizeP, numOfCubesOri, numOfParticles;

    int   *cubeId, *cubeMapping;
		int   *tmpNxt, *tmpFree;

		int   *edgesSrc, *edgesDes;
    float *edgesWeight;
		int   *edgeStartOfCubes, *edgeSizeOfCubes;

    int *leafParent, *leafChildS, *leafChildE, *leafSibling;
    int *leafI, *leafCount;

    float *leafValue;
    float *leafX, *leafY, *leafZ;
    float *leafVX, *leafVY, *leafVZ;

    __host__ __device__
    combineMergeTrees(int *cubeMapping, int *cubeId, int *tmpNxt, int *tmpFree,
        int *leafParent, int *leafChildS, int *leafChildE, int *leafSibling,
        int *leafI, float *leafValue, int *leafCount,
        float *leafX, float *leafY, float *leafZ,
        float *leafVX, float *leafVY, float *leafVZ,
        int *edgesSrc, int *edgesDes, float *edgesWeight,
        int *edgeStartOfCubes, int *edgeSizeOfCubes,
				float min_ll, int sizeP, int numOfCubesOri, int numOfParticles) :
        cubeMapping(cubeMapping), cubeId(cubeId), tmpNxt(tmpNxt), tmpFree(tmpFree),
        leafParent(leafParent), leafChildS(leafChildS), leafChildE(leafChildE), leafSibling(leafSibling),
        leafI(leafI), leafValue(leafValue), leafCount(leafCount),
        leafX(leafX), leafY(leafY), leafZ(leafZ),
        leafVX(leafVX), leafVY(leafVY), leafVZ(leafVZ),
        edgesSrc(edgesSrc), edgesDes(edgesDes), edgesWeight(edgesWeight),
        edgeStartOfCubes(edgeStartOfCubes), edgeSizeOfCubes(edgeSizeOfCubes),
				min_ll(min_ll), sizeP(sizeP), numOfCubesOri(numOfCubesOri), numOfParticles(numOfParticles) {}

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

					int src = eSrc;
					int des = eDes;

					float weight = (eWeight < min_ll) ? min_ll : eWeight;

					// find the src & des nodes just below the required weight
					while(leafParent[src]!=-1 && leafValue[leafParent[src]]<=weight) src = leafParent[src];
					while(leafParent[des]!=-1 && leafValue[leafParent[des]]<=weight) des = leafParent[des];

					// if src & des already have the same halo id, do NOT do anything
          if(leafI[src]==leafI[des]) continue;

          int srcCount = leafCount[src];
          int desCount = leafCount[des];

          float srcX = leafX[src]; float desX = leafX[des]; float srcVX = leafVX[src]; float desVX = leafVX[des];
          float srcY = leafY[src]; float desY = leafY[des]; float srcVY = leafVY[src]; float desVY = leafVY[des];
          float srcZ = leafZ[src]; float desZ = leafZ[des]; float srcVZ = leafVZ[src]; float desVZ = leafVZ[des];

          // get the original parents of src & des nodes
          int srcTmp = leafParent[src];
          int desTmp = leafParent[des];

          // remove the src & des from the child list of their parents
          if(srcTmp!=-1)
          {
            int child = leafChildS[srcTmp];
            if(child!=-1 && leafValue[child]==leafValue[src] && leafI[child]==leafI[src])
              leafChildS[srcTmp] = leafSibling[child];
            else
            {
              while(child!=-1 && leafSibling[child]!=-1)
              {
                if(leafValue[leafSibling[child]]==leafValue[src] && leafI[leafSibling[child]]==leafI[src])
                { leafSibling[child] = leafSibling[leafSibling[child]];   break; }
                child = leafSibling[child];
              }

              if(child!=-1 && leafSibling[child]==-1) leafChildE[srcTmp] = child;
            }

            if(leafChildS[srcTmp]==-1) leafChildE[srcTmp] = -1;
          }
          leafParent[src] = -1; leafSibling[src] = -1;

          if(desTmp!=-1)
          {
            int child = leafChildS[desTmp];
            if(child!=-1 && leafValue[child]==leafValue[des] && leafI[child]==leafI[des])
              leafChildS[desTmp] = leafSibling[child];
            else
            {
              while(child!=-1 && leafSibling[child]!=-1)
              {
                if(leafValue[leafSibling[child]]==leafValue[des] && leafI[leafSibling[child]]==leafI[des])
                { leafSibling[child] = leafSibling[leafSibling[child]];   break; }
                child = leafSibling[child];
              }

              if(child!=-1 && leafSibling[child]==-1) leafChildE[desTmp] = child;
            }

            if(leafChildS[desTmp]==-1) leafChildE[desTmp] = -1;
          }
          leafParent[des] = -1; leafSibling[des] = -1;



          // set n node
          int n;
          bool freeDes=false;
          if(leafValue[src]==weight && leafValue[des]==weight) // merge src & des, free des node, set n to src, then connect their children & fix the loop
          {
            n = src;
            int child = leafChildS[des];
            while(child!=-1) { leafParent[child] = n; child = leafSibling[child]; }

            if(leafChildE[n]!=-1)  leafSibling[leafChildE[n]] = leafChildS[des];
            else  leafChildS[n] = leafChildS[des];
            leafChildE[n] = leafChildE[des];
            freeDes = true;
          }
          else if(leafValue[src]==weight) // set des node's parent to be src, set n to src, then fix the loop
          {
            n = src;  leafSibling[leafChildE[n]] = des;   leafChildE[n] = des;    leafParent[des] = n;
          }
          else if(leafValue[des]==weight) // set src node's parent to be des, set n to des, then fix the loop
          {
            n = des;  leafSibling[leafChildE[n]] = src;   leafChildE[n] = src;    leafParent[src] = n;
          }
          else if(leafValue[src]!=weight && leafValue[des]!=weight) // create a new node, set this as parent of both src & des, then fix the loop
          {
            if(tmpNxt[cubeStart]!=-1)
            {
              n = tmpNxt[cubeStart] + numOfParticles;
              int tmpVal = tmpFree[tmpNxt[cubeStart]];
              tmpFree[tmpNxt[cubeStart]] = -2;
              tmpNxt[cubeStart] = tmpVal;

              leafChildS[n] = src;  leafChildE[n] = des;
              leafParent[src] = n;  leafParent[des] = n;
              leafSibling[src] = des;
            }
            #if THRUST_DEVICE_BACKEND != THRUST_DEVICE_BACKEND_CUDA
            else
              std::cout << "***no Free item .... this shouldnt happen*** " << cubeStart << std::endl;
            #endif
          }

          leafValue[n] = weight;
          leafCount[n] = leafCount[src] + leafCount[des];
          leafX[n] = leafX[src]+leafX[des];   leafVX[n] = leafVX[src]+leafVX[des];
          leafY[n] = leafY[src]+leafY[des];   leafVY[n] = leafVY[src]+leafVY[des];
          leafZ[n] = leafZ[src]+leafZ[des];   leafVZ[n] = leafVZ[src]+leafVZ[des];
          leafI[n] = (leafI[src] < leafI[des]) ? leafI[src] : leafI[des];

          if(freeDes && des>=numOfParticles)
          {
            // free des node
            int tmpVal = tmpNxt[cubeStart];
            tmpNxt[cubeStart] = des-numOfParticles;
            tmpFree[tmpNxt[cubeStart]] = tmpVal;

            leafI[des] = -1;
            leafValue[des] = 0.0f;
            leafCount[des] = 0;
            leafX[des] = 0; leafVX[des] = 0;
            leafY[des] = 0; leafVX[des] = 0;
            leafZ[des] = 0; leafVX[des] = 0;
            leafParent[des] = -1;
            leafChildS[des] = -1;
            leafChildE[des] = -1;
            leafSibling[des] = -1;
          }



          bool done = false;
          while(srcTmp!=-1 && desTmp!=-1)
          {
            if(leafValue[srcTmp] < leafValue[desTmp])
            {
              leafParent[n] = srcTmp;

              if(leafChildE[srcTmp]!=-1) leafSibling[leafChildE[srcTmp]] = n;
              else leafChildS[srcTmp] = n;
              leafChildE[srcTmp] = n;

              leafI[srcTmp] = (leafI[srcTmp]<leafI[n]) ? leafI[srcTmp] : leafI[n];
              srcCount = leafCount[srcTmp];
              leafCount[srcTmp] += desCount;
              srcX = leafX[srcTmp];   srcVX = leafVX[srcTmp];
              srcY = leafY[srcTmp];   srcVY = leafVY[srcTmp];
              srcZ = leafZ[srcTmp];   srcVZ = leafVZ[srcTmp];
              leafX[srcTmp] += desX;  leafVX[srcTmp] += desVX;
              leafY[srcTmp] += desY;  leafVY[srcTmp] += desVY;
              leafZ[srcTmp] += desZ;  leafVZ[srcTmp] += desVZ;

              n = srcTmp;
              srcTmp = leafParent[srcTmp];

              if(srcTmp!=-1)
              {
                int child = leafChildS[srcTmp];
                if(child!=-1 && leafValue[child]==leafValue[n] && leafI[child]==leafI[n])
                  leafChildS[srcTmp] = leafSibling[child];
                else
                {
                  while(child!=-1 && leafSibling[child]!=-1)
                  {
                    if(leafValue[leafSibling[child]]==leafValue[n] && leafI[leafSibling[child]]==leafI[n])
                    { leafSibling[child] = leafSibling[leafSibling[child]]; break; }
                    child = leafSibling[child];
                  }

                  if(child!=-1 && leafSibling[child]==-1) leafChildE[srcTmp] = child;
                }

                if(leafChildS[srcTmp]==-1) leafChildE[srcTmp] = -1;
              }
              leafParent[n] = -1; leafSibling[n] = -1;
            }
            else if(leafValue[srcTmp] > leafValue[desTmp])
            {
              leafParent[n] = desTmp;

              if(leafChildE[desTmp]!=-1) leafSibling[leafChildE[desTmp]] = n;
              else leafChildS[desTmp] = n;
              leafChildE[desTmp] = n;

              leafI[desTmp] = (leafI[desTmp]<leafI[n]) ? leafI[desTmp] : leafI[n];
              desCount = leafCount[desTmp];
              leafCount[desTmp] += srcCount;
              desX = leafX[desTmp];   desVX = leafVX[desTmp];
              desY = leafY[desTmp];   desVY = leafVY[desTmp];
              desZ = leafZ[desTmp];   desVZ = leafVZ[desTmp];
              leafX[desTmp] += srcX;  leafVX[desTmp] += srcVX;
              leafY[desTmp] += srcY;  leafVY[desTmp] += srcVY;
              leafZ[desTmp] += srcZ;  leafVZ[desTmp] += srcVZ;

              n = desTmp;
              desTmp = leafParent[desTmp];

              if(desTmp!=-1)
              {
                int child = leafChildS[desTmp];
                if(child!=-1 && leafValue[child]==leafValue[n] && leafI[child]==leafI[n])
                  leafChildS[desTmp] = leafSibling[child];
                else
                {
                  while(child!=-1 && leafSibling[child]!=-1)
                  {
                    if(leafValue[leafSibling[child]]==leafValue[n] && leafI[leafSibling[child]]==leafI[n])
                    { leafSibling[child] = leafSibling[leafSibling[child]]; break; }
                    child = leafSibling[child];
                  }

                  if(child!=-1 && leafSibling[child]==-1) leafChildE[desTmp] = child;
                }

                if(leafChildS[desTmp]==-1) leafChildE[desTmp] = -1;
              }
              leafParent[n] = -1; leafSibling[n] = -1;
            }
            else if(leafValue[srcTmp] == leafValue[desTmp])
            {
              if(leafI[srcTmp] != leafI[desTmp]) // combine srcTmp & desTmp
              {
                int child = leafChildS[desTmp];
                while(child!=-1) { leafParent[child] = srcTmp;  child = leafSibling[child]; }

                if(leafChildE[srcTmp]!=-1) leafSibling[leafChildE[srcTmp]] = leafChildS[desTmp];
                else leafChildS[srcTmp] = leafChildS[desTmp];
                leafChildE[srcTmp] = leafChildE[desTmp];

                leafI[srcTmp] = (leafI[srcTmp]<leafI[desTmp]) ? leafI[srcTmp] : leafI[desTmp];
                leafCount[srcTmp] += leafCount[desTmp];
                leafX[srcTmp] += leafX[desTmp];  leafVX[srcTmp] += leafVX[desTmp];
                leafY[srcTmp] += leafY[desTmp];  leafVY[srcTmp] += leafVY[desTmp];
                leafZ[srcTmp] += leafZ[desTmp];  leafVZ[srcTmp] += leafVZ[desTmp];
              }

              if(leafChildS[srcTmp]==-1 && leafChildE[srcTmp]==-1)
              {
                if(leafParent[srcTmp]!=-1)
                {
                  int child = leafChildS[leafParent[srcTmp]];
                  if(child!=-1 && leafValue[child]==leafValue[srcTmp] && leafI[child]==leafI[srcTmp])
                    leafChildS[leafParent[srcTmp]] = leafSibling[child];
                  else
                  {
                    while(child!=-1 && leafSibling[child]!=-1)
                    {
                      if(leafValue[leafSibling[child]]==leafValue[srcTmp] && leafI[leafSibling[child]]==leafI[srcTmp])
                      { leafSibling[child] = leafSibling[leafSibling[child]]; break; }
                      child = leafSibling[child];
                    }

                    if(child!=-1 && leafSibling[child]==-1) leafChildE[leafParent[srcTmp]] = child;
                  }

                  if(leafChildS[leafParent[srcTmp]]==1) leafChildE[leafParent[srcTmp]] = -1;
                }

                int tmp = leafParent[srcTmp];
                if(srcTmp>=numOfParticles)
                {
                  // free srcTmp node
                  int tmpVal = tmpNxt[cubeStart];
                  tmpNxt[cubeStart] = srcTmp-numOfParticles;
                  tmpFree[tmpNxt[cubeStart]] = tmpVal;

                  leafI[srcTmp] = -1;
                  leafValue[srcTmp] = 0.0f;
                  leafCount[srcTmp] = 0;
                  leafX[srcTmp] = 0;   leafVX[srcTmp] = 0;
                  leafY[srcTmp] = 0;   leafVX[srcTmp] = 0;
                  leafZ[srcTmp] = 0;   leafVX[srcTmp] = 0;
                  leafParent[srcTmp] = -1;
                  leafChildS[srcTmp] = -1;
                  leafChildE[srcTmp] = -1;
                  leafSibling[srcTmp] = -1;
                }
                srcTmp = tmp;
              }

              if(srcTmp!=-1)
              {
                leafParent[n] = srcTmp;

                if(leafChildE[srcTmp]!=-1) leafSibling[leafChildE[srcTmp]] = n;
                else  leafChildS[srcTmp] = n;
                leafChildE[srcTmp] = n;

                leafI[srcTmp] = (leafI[srcTmp]<leafI[n]) ? leafI[srcTmp] : leafI[n];
              }
              else  leafParent[n] = -1;

              done = true;
              break;
            }
          }



          if(!done && srcTmp!=-1)
          {
            leafParent[n] = srcTmp;

            if(leafChildE[srcTmp])  leafSibling[leafChildE[srcTmp]] = n;
            else  leafChildS[srcTmp] = n;
            leafChildE[srcTmp] = n;

            leafI[srcTmp] = (leafI[srcTmp]<leafI[n]) ? leafI[srcTmp] : leafI[n];
            srcCount = leafCount[srcTmp];
            leafCount[srcTmp] += desCount;
            srcX = leafX[srcTmp];   srcVX = leafVX[srcTmp];
            srcY = leafY[srcTmp];   srcVY = leafVY[srcTmp];
            srcZ = leafZ[srcTmp];   srcVZ = leafVZ[srcTmp];
            leafX[srcTmp] += desX;  leafVX[srcTmp] += desVX;
            leafY[srcTmp] += desY;  leafVY[srcTmp] += desVY;
            leafZ[srcTmp] += desZ;  leafVZ[srcTmp] += desVZ;

            n = srcTmp;
            srcTmp = leafParent[srcTmp];
          }
          while(!done && srcTmp!=-1)
          {
            leafI[srcTmp] = (leafI[srcTmp]<leafI[n]) ? leafI[srcTmp] : leafI[n];
            srcCount = leafCount[srcTmp];
            leafCount[srcTmp] += desCount;
            srcX = leafX[srcTmp];   srcVX = leafVX[srcTmp];
            srcY = leafY[srcTmp];   srcVY = leafVY[srcTmp];
            srcZ = leafZ[srcTmp];   srcVZ = leafVZ[srcTmp];
            leafX[srcTmp] += desX;  leafVX[srcTmp] += desVX;
            leafY[srcTmp] += desY;  leafVY[srcTmp] += desVY;
            leafZ[srcTmp] += desZ;  leafVZ[srcTmp] += desVZ;

            n = srcTmp;
            srcTmp = leafParent[srcTmp];
          }


          if(!done && desTmp!=-1)
          {
            leafParent[n] = desTmp;

            if(leafChildE[desTmp])  leafSibling[leafChildE[desTmp]] = n;
            else  leafChildS[desTmp] = n;
            leafChildE[desTmp] = n;

            leafI[desTmp] = (leafI[desTmp]<leafI[n]) ? leafI[desTmp] : leafI[n];
            desCount = leafCount[desTmp];
            leafCount[desTmp] += srcCount;
            desX = leafX[desTmp];   desVX = leafVX[desTmp];
            desY = leafY[desTmp];   desVY = leafVY[desTmp];
            desZ = leafZ[desTmp];   desVZ = leafVZ[desTmp];
            leafX[desTmp] += srcX;  leafVX[desTmp] += srcVX;
            leafY[desTmp] += srcY;  leafVY[desTmp] += srcVY;
            leafZ[desTmp] += srcZ;  leafVZ[desTmp] += srcVZ;

            n = desTmp;
            desTmp = leafParent[desTmp];
          }
          while(!done && desTmp!=-1)
          {
            leafI[desTmp] = (leafI[desTmp]<leafI[n]) ? leafI[desTmp] : leafI[n];
            desCount = leafCount[desTmp];
            leafCount[desTmp] += srcCount;
            desX = leafX[desTmp];   desVX = leafVX[desTmp];
            desY = leafY[desTmp];   desVY = leafVY[desTmp];
            desZ = leafZ[desTmp];   desVZ = leafVZ[desTmp];
            leafX[desTmp] += srcX;  leafVX[desTmp] += srcVX;
            leafY[desTmp] += srcY;  leafVY[desTmp] += srcVY;
            leafZ[desTmp] += srcZ;  leafVZ[desTmp] += srcVZ;

            n = desTmp;
            desTmp = leafParent[desTmp];
          }

					//-----------------------------------------------------

				}

				edgeSizeOfCubes[k] = size;
			}
		}		
	};	
};

}

#endif


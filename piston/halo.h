#ifndef HALO_H_
#define HALO_H_

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
#include <thrust/replace.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/merge.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <piston/kd.h>

#include <sys/time.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <cmath>

namespace piston
{

class halo
{
public:

  // definitions
  typedef thrust::device_vector<float>::iterator FloatIterator;
  typedef thrust::device_vector<int>::iterator   IntIterator;

  typedef thrust::tuple<float, float> Float2;
  typedef thrust::tuple<float, float, float> Float3;
  typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> Float3TupleIterator;
  typedef thrust::zip_iterator<Float3TupleIterator> Float3zipIterator;

  typedef Float3 ParticleTuple;
  typedef Float3TupleIterator ParticleTupleIterator;
  typedef Float3zipIterator ParticleTupleZipIterator;

  typedef thrust::counting_iterator<int> CountingIterator;
  typedef thrust::constant_iterator<int> ConstantIterator;

  // inputs
  float linkLength;     // linking length used to link two particles
  int   particleSize;   // number of particles in a halo

	int   n;		 					// if you want a fraction of the file to load, set this.. 1/n
  float xscal; 					// scale factor for linking length
  int   numOfParticles; // total number of particles

  int   numOfHalos;     		// total number of halos
  int   numOfHaloParticles; // total number of particles in halos

	float lBoundX, lBoundY, lBoundZ; // lower bound of the entire space
	float uBoundX, uBoundY, uBoundZ; // upper bound of the entire space

	thrust::device_vector<int>    haloIndex; // halo indices for each particle

	// stores all node details of the merge tree
  thrust::device_vector<int>    leafI, leafM;							// index & mass for each leaf
  thrust::device_vector<float>  leafX, leafY, leafZ;	    // positions for each leaf
  thrust::device_vector<float>  leafVX, leafVY, leafVZ;	  // velocities for each leaf
  thrust::device_vector<float>  leafValue;                // value for each leaf
  thrust::device_vector<int>    leafCount;                // count for each leaf
  thrust::device_vector<int>    leafParent, leafParentS;
  thrust::device_vector<int>    leafChildS, leafChildE, leafSibling;

	// stores stats of each halos
  thrust::device_vector<int>    haloIndexUnique;            					 // unique halo indexes
	thrust::device_vector<int>    haloCount;														 // number of particles in each halo
  thrust::device_vector<float>  haloX, haloY, haloZ; 									 // positions for each halo
  thrust::device_vector<float>  haloVX, haloVY, haloVZ;    						 // velocities for each halo
  thrust::device_vector<float>  haloColorsR, haloColorsG, haloColorsB; // colors for each halo

	// stores details about just the particles which belong to halos
	thrust::device_vector<int>    haloIndex_f;	  			
	thrust::device_vector<float>  inputM_f;
  thrust::device_vector<float>  inputX_f, inputY_f, inputZ_f;
  thrust::device_vector<float>  inputVX_f, inputVY_f, inputVZ_f;

  // variables needed to create random numbers
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> u01;

  halo(std::string filename="", std::string format=".cosmo", int n=1, int np=1, float rL=-1)
  {
    u01 = thrust::uniform_real_distribution<float>(0.0f, 1.0f);

		this->n = n;

		// scale amount for particles
	  if(rL==-1) xscal = 1;
	  else       xscal = rL / (1.0*np);

    if(!readHaloFile(filename, format))
    {			
//			generateUniformData();		// generate uniformly spaced points		
//			generateNonUniformData(); // generate nearby points to real data

			numOfParticles = 8;
			leafX = thrust::host_vector<float>(numOfParticles);
			leafY = thrust::host_vector<float>(numOfParticles);
			leafZ = thrust::host_vector<float>(numOfParticles);
			leafI = thrust::host_vector<float>(numOfParticles);

			leafX[0] = 1.0;	leafY[0] = 1.0;	leafZ[0] = 0.0; leafI[0] = 7;
			leafX[1] = 0.0;	leafY[1] = 4.0;	leafZ[1] = 0.0; leafI[1] = 6;
			leafX[2] = 8.0;	leafY[2] = 3.0;	leafZ[2] = 0.0; leafI[2] = 5;
			leafX[3] = 2.0;	leafY[3] = 6.0;	leafZ[3] = 0.0; leafI[3] = 4;
			leafX[4] = 4.0;	leafY[4] = 2.0;	leafZ[4] = 0.0; leafI[4] = 3;
			leafX[5] = 5.0;	leafY[5] = 5.0;	leafZ[5] = 0.0; leafI[5] = 2;
			leafX[6] = 3.0;	leafY[6] = 4.0;	leafZ[6] = 0.0; leafI[6] = 1;
			leafX[7] = 3.5;	leafY[7] = 4.5;	leafZ[7] = 0.0; leafI[7] = 0;

			std::cout << "Test data loaded \n";
    }

    getBounds();

    std::cout << "numOfParticles : " << numOfParticles << " \n";
  }

  virtual void operator()(float linkLength , int  particleSize) {}


  //------- read input files

  // read input file - currently can read a .cosmo, .hcosmo or .csv file
  // .csv file - when you have a big data file and you want a piece of it, load it in VTK and slice it and save it as .csv file, within this function it will rewrite the date to .cosmo format
  bool readHaloFile(std::string filename, std::string format)
  {
    // check filename
    if(filename == "") { std::cout << "no input file specified \n"; return false; }
		if(format == "cosmo")  return readCosmoFile(filename, format);
		if(format == "hcosmo") return readHCosmoFile(filename, format);
    if(format == "csv")    return readCsvFile(filename, format);   

    return false;
  }

  // read a .cosmo file and load the data to leafX, leafY & leafZ
	bool readCosmoFile(std::string filename, std::string format)
	{
		// open .cosmo file
		std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
		if (!myfile->is_open()) { std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

		std::cout << filename << std::endl;

		int nfloat = 7, nint = 1;

		// declare temporary read buffers
		float fBlock[nfloat];
		int   iBlock[nint];
		
		// compute the number of particles
		myfile->seekg(0L, std::ios::end);
		numOfParticles = myfile->tellg() / (nfloat*sizeof(float)+nint*sizeof(int)); // get particle size in file

		// get the fraction wanted
		numOfParticles = numOfParticles / n; 

		// resize
		leafX.resize(numOfParticles);	      leafY.resize(numOfParticles);	   leafZ.resize(numOfParticles);
		leafVX.resize(numOfParticles);	    leafVY.resize(numOfParticles);	 leafVZ.resize(numOfParticles);
		leafM.resize(numOfParticles);       leafI.resize(numOfParticles);

		// rewind file to beginning for particle reads
		myfile->seekg(0L, std::ios::beg);

		thrust::host_vector<float> leafX_h(numOfParticles);
		thrust::host_vector<float> leafY_h(numOfParticles);
		thrust::host_vector<float> leafZ_h(numOfParticles);
		thrust::host_vector<float> leafVX_h(numOfParticles);
		thrust::host_vector<float> leafVY_h(numOfParticles);
		thrust::host_vector<float> leafVZ_h(numOfParticles);
		thrust::host_vector<float> leafM_h(numOfParticles);
		thrust::host_vector<float> leafI_h(numOfParticles);

		for (int i=0; i<numOfParticles; i++)
		{
			// Set file pointer to the requested particle
			myfile->read(reinterpret_cast<char*>(fBlock), nfloat * sizeof(float));
			if (myfile->gcount() != (int)(nfloat * sizeof(float))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			myfile->read(reinterpret_cast<char*>(iBlock), nint * sizeof(int));
			if (myfile->gcount() != (int)(nint * sizeof(int))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			leafX_h[i] = fBlock[0];	leafVX_h[i] = fBlock[1];
			leafY_h[i] = fBlock[2];	leafVY_h[i] = fBlock[3];
			leafZ_h[i] = fBlock[4];	leafVZ_h[i] = fBlock[5];
			leafM_h[i] = fBlock[6];

			leafI_h[i] = iBlock[0];
		}
		myfile->close();

		thrust::copy(leafX_h.begin(),  leafX_h.end(),  leafX.begin());
		thrust::copy(leafY_h.begin(),  leafY_h.end(),  leafY.begin());
		thrust::copy(leafZ_h.begin(),  leafZ_h.end(),  leafZ.begin());
		thrust::copy(leafVX_h.begin(), leafVX_h.end(), leafVX.begin());
		thrust::copy(leafVY_h.begin(), leafVY_h.end(), leafVY.begin());
		thrust::copy(leafVZ_h.begin(), leafVZ_h.end(), leafVZ.begin());
		thrust::copy(leafM_h.begin(),  leafM_h.end(),  leafM.begin());
		thrust::copy(leafI_h.begin(),  leafI_h.end(),  leafI.begin());

		return true;
	}

	// read a .hcosmo file and load the data to leafX, leafY & leafZ
	bool readHCosmoFile(std::string filename, std::string format)
	{
		// open .hcosmo file
		std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
		if (!myfile->is_open()) { std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }
		
		int nfloat = 7, nint = 1;

		// declare temporary read buffers
		float    fBlock[nfloat];
		long int iBlock[nint];
		int      iBlockTmp[nint];

		// compute the number of particles
		myfile->seekg(0L, std::ios::end);
		numOfParticles = myfile->tellg() / (nfloat*sizeof(float)+nint*sizeof(long int)); // get particle size in file

		// get the fraction wanted
 		numOfParticles = numOfParticles / n;

		// rewind file to beginning for particle reads

 		myfile->seekg(0L, std::ios::beg);

		// Create one for output file
		std::ofstream *outStream = new std::ofstream();
		outStream->open((filename+".cosmo").c_str(), std::ios::out|std::ios::binary);
		for (int i=0; i<numOfParticles; i++)
		{
			// Set file pointer to the requested particle
			myfile->read(reinterpret_cast<char*>(fBlock), nfloat * sizeof(float));
			if (myfile->gcount() != (int)(nfloat * sizeof(float))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			myfile->read(reinterpret_cast<char*>(iBlock), nint * sizeof(long int));
			if (myfile->gcount() != (int)(nint * sizeof(long int))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			iBlockTmp[0] = (int) iBlock[0];

			// write to file
		  outStream->write(reinterpret_cast<const char*>(fBlock), nfloat * sizeof(float));
		  outStream->write(reinterpret_cast<const char*>(iBlockTmp), nint * sizeof(int));
		}
		outStream->close();
		myfile->close();

    {
      std::string filename = "/home/wathsy/Desktop/HaloFinder/data/Small/output/m000.499.allparticles.ascii";
      std::ifstream *myfile = new std::ifstream(filename.c_str(), std::ios::in);
      if (!myfile->is_open()) { std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

      std::map<int,int> vec;

      std::string line;
      if (myfile->is_open())
      {
        int count = 0;
        while(count++<7)
          getline(*myfile,line);

        count = 0;
        while(getline(*myfile,line))
        {
          count++;

          float x, y, z, vx, vy, vz;
          int id, fof_halo_tag;

          std::string sub;
          std::istringstream iss(line);

          iss >> sub; x = atof(sub.c_str());  iss >> sub; y = atof(sub.c_str());  iss >> sub; z = atof(sub.c_str());
          iss >> sub; vx = atof(sub.c_str()); iss >> sub; vy = atof(sub.c_str()); iss >> sub; vz = atof(sub.c_str());
          iss >> sub; id = atoi(sub.c_str()); iss >> sub; fof_halo_tag = atoi(sub.c_str());

          std::map<int,int>::iterator tIt = vec.find(id);
          if(tIt == vec.end())
            vec.insert(std::map<int,int>::value_type(id,fof_halo_tag));
          else
            std::cout << (tIt->first) << " " << id << " - " << (tIt->second) << " " << fof_halo_tag << std::endl;
        }

        std::cout << count-vec.size() << " out of " << count << " is duplicate. Rest is " << vec.size() << std::endl;

        myfile->close();
      }
    }

		n = 1;	

		// read .cosmo file 
		readCosmoFile(filename, "cosmo"); 

		return true;		
	}

  // read a .csv file and write it to .cosmo format & read it from that
  bool readCsvFile(std::string filename, std::string format)
  {
		// open .csv file
		std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
		if (!myfile->is_open()) { std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

		std::string line;
		getline(*myfile,line);

		int nfloat = 7, nint = 1;

		// declare temporary read buffers
		float fBlock[nfloat];
		int   iBlock[nint];

		// Create one for output file
		std::ofstream *outStream = new std::ofstream();
		outStream->open((filename+".cosmo").c_str(), std::ios::out|std::ios::binary);
		while(!myfile->eof())
		{
			getline(*myfile,line);
			if(line=="") continue;

			int i = 0;
			strtok((char*)line.c_str(), ",");		
			while(++i<4) strtok (NULL, ",");
			iBlock[0] = atoi(strtok(NULL, ","));

			while(++i<7) strtok (NULL, ",");
			fBlock[0] = atof(strtok(NULL, ","));	fBlock[1] = 0;
			fBlock[2] = atof(strtok(NULL, ","));	fBlock[3] = 0;
			fBlock[4] = atof(strtok(NULL, ","));	fBlock[5] = 0;
			fBlock[6] = 1;

			// write to file
		  outStream->write(reinterpret_cast<const char*>(fBlock), nfloat * sizeof(float));
		  outStream->write(reinterpret_cast<const char*>(iBlock), nint * sizeof(int));
		}	
		outStream->close();
		myfile->close();

		// read .cosmo file
		readCosmoFile(filename, "cosmo");  

    return true;
  }


  //------- generate data

	// generate uniform data & write it to .cosmo format & then read from that
	void generateUniformData()
	{
		// set bounds
	  lBoundX = 0.1;   lBoundY = 0.1;   lBoundZ = 0.1;
	  uBoundX = 11.2;  uBoundY = 11.2;  uBoundZ = 11.2;

		int nX=16, nY=16, nZ=16;
		numOfParticles = nX*nY*nZ;		

		double startX=lBoundX;	double stepX=(uBoundX-lBoundX)/(nX-1);
		double startY=lBoundY;	double stepY=(uBoundY-lBoundY)/(nY-1);
		double startZ=lBoundZ; 	double stepZ=(uBoundZ-lBoundZ)/(nZ-1);

		int nfloat = 7, nint = 1;

		// declare temporary read buffers
		float fBlock[nfloat];
		int   iBlock[nint];

		int num = 0;

		// Create one for output file
		std::ofstream *outStream = new std::ofstream();
		outStream->open((convertInt(numOfParticles)+".cosmo").c_str(), std::ios::out|std::ios::binary);
		for(int xx=0; xx<nX; xx++)
		{
			for(int yy=0; yy<nY; yy++)
			{
				for(int zz=0; zz<nZ; zz++)
				{
					float x = (float)(startX+stepX*xx);	
					float y = (float)(startY+stepY*yy);	
					float z = (float)(startZ+stepZ*zz);

					fBlock[0] = x;	fBlock[1] = 0;
					fBlock[2] = y;	fBlock[3] = 0;
					fBlock[4] = z;	fBlock[5] = 0;
					fBlock[6] = 1;

					iBlock[0] = num++;

					// write to file
					outStream->write(reinterpret_cast<const char*>(fBlock), nfloat * sizeof(float));
					outStream->write(reinterpret_cast<const char*>(iBlock), nint * sizeof(int));
				}
			}
		}			 
		outStream->close();

		std::cout << "UniformData generated \n";
		
		// read .cosmo file 
		readCosmoFile(convertInt(numOfParticles), "cosmo");  
	}

	// generate NON uniform data & write it to .cosmo format & then read from that
	void generateNonUniformData()
	{
		// variables needed to create random numbers
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> u;
		u = thrust::uniform_real_distribution<float>(-0.5f, 0.5f);

		// set the input file information
		readHaloFile("/home/wathsy/Desktop/HaloFinder/data/sub-24474", "cosmo");

		int numPoints = 1;

		int nfloat = 7, nint = 1;

		// declare temporary read buffers
		float fBlock[nfloat];
		int   iBlock[nint];
	
		int num = 0;

		// Create one for output file
		std::ofstream *outStream = new std::ofstream();
		outStream->open((convertInt(numOfParticles)+".cosmo").c_str(), std::ios::out|std::ios::binary);
		for(int i=0; i<numOfParticles; i++)
		{
			for(int j=1; j<numPoints; j++)
			{
				float x = (float)(leafX[i] + u(rng));
				float y = (float)(leafY[i] + u(rng));
				float z = (float)(leafZ[i] + u(rng));

				while(x<lBoundX || x>uBoundX) x = (float)(leafX[i] + u(rng));
				while(y<lBoundY || y>uBoundY) y = (float)(leafY[i] + u(rng));
				while(z<lBoundZ || z>uBoundZ) z = (float)(leafZ[i] + u(rng));

				fBlock[0] = x;	fBlock[1] = leafVX[i];
				fBlock[2] = y;	fBlock[3] = leafVY[i];
				fBlock[4] = z;	fBlock[5] = leafVZ[i];
				fBlock[6] = leafM[i];

				iBlock[0] = num++;

				// write to file
				outStream->write(reinterpret_cast<const char*>(fBlock), nfloat * sizeof(float));
				outStream->write(reinterpret_cast<const char*>(iBlock), nint * sizeof(int));
			}
		}
		outStream->close();

		numOfParticles *= numPoints; // set the new numOfParticles

		std::cout << "Non UniformData generated \n";

		// read .cosmo file
		readCosmoFile(convertInt(numOfParticles), "cosmo"); 
	}	


	//------- other functions

	// write halo id details for each particle in to a .txt file
	void writeHaloResults()
	{
		// copy halo info of particles
		thrust::device_vector<int> a;		
		a.resize(numOfParticles);
		thrust::copy(haloIndex.begin(), haloIndex.end(), a.begin());

		//write to file
		std::string filename = convertInt(numOfParticles) + "_MTree.txt";
		std::ofstream out(filename.c_str(), std::ofstream::out); 
		for(int i=0; i<numOfParticles; i++)
		{
			if(i%15==0) out << "\n";
			out << a[i] << " ";
		}
		out.close();
	}

	// convert int to string
	std::string convertInt(int number)
	{
		std::stringstream ss; //create a stringstream
		ss << number;					//add number to the stream
		return ss.str();			//return a string with the contents of the stream
	}

	// get lower & upper bounds of the entire space
	void getBounds()
	{
		// get bounds of the space
    typedef thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> result_type;
    result_type result1 = thrust::minmax_element(leafX.begin(), leafX.end());
    result_type result2 = thrust::minmax_element(leafY.begin(), leafY.end());
    result_type result3 = thrust::minmax_element(leafZ.begin(), leafZ.end());

    // set bounds
    lBoundX = *result1.first;   lBoundY = *result2.first;   lBoundZ = *result3.first;
    uBoundX = *result1.second;  uBoundY = *result2.second;  uBoundZ = *result3.second;
	}

	// get the index in haloIndexUnique for a halo i
  int getHaloInd(int i, bool useF=false)
  {
		int id = (useF) ? haloIndex_f[i] : haloIndex[i];
    IntIterator ite = thrust::find(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfHalos, id);		
    return (ite!=haloIndexUnique.begin()+numOfHalos) ? ite - haloIndexUnique.begin() : -1;
  }

  // return a vector with N random values in the range [min,max)
  thrust::host_vector<float> random_vector(const size_t N, float max, float min)
  {
    thrust::host_vector<float> tmp(N);
    for(size_t i = 0; i < N; i++) tmp[i] = (u01(rng)*(max-min) + min);
    return tmp;
  }

  // clear vectors & variables
  void clear()
  {
    haloIndex.clear();
    haloIndexUnique.clear();

    haloColorsR.clear();
    haloColorsG.clear();
    haloColorsB.clear();

    numOfHalos = 0;
    
    haloIndex.resize(numOfParticles);
		thrust::copy(CountingIterator(0), CountingIterator(0)+numOfParticles, haloIndex.begin());
  }

  // set colors to halos
  void setColors()
  {
    // set color range
    float minrangeC = 0.1;
    float maxrangeC = 1;

    // for each halo, set unique colors
    u01 = thrust::uniform_real_distribution<float>(0.0f, 1.0f);
    haloColorsR = random_vector(numOfHalos, maxrangeC, minrangeC);
    haloColorsG = random_vector(numOfHalos, maxrangeC, minrangeC);
    haloColorsB = random_vector(numOfHalos, maxrangeC, minrangeC);
  }

  // get start of vertices
  Float3zipIterator vertices_begin()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin()));
  }

  // get end of vertices
  Float3zipIterator vertices_end()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(leafX.end(), leafY.end(), leafZ.end()));
  }

	// get start of vertices_f 
  Float3zipIterator vertices_begin_f()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(inputX_f.begin(), inputY_f.begin(), inputZ_f.begin()));
  }

  // get end of vertices_f
  Float3zipIterator vertices_end_f()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(inputX_f.end(), inputY_f.end(), inputZ_f.end()));
  }

  // get start of halos
  IntIterator halos_begin()
  {
    return haloIndex.begin();
  }

  // get end of halos
  IntIterator halos_end()
  {
    return haloIndex.end();
  }

	// get start of halos_f
  IntIterator halos_begin_f()
  {
    return haloIndex_f.begin();
  }

  // get end of halos_f
  IntIterator halos_end_f()
  {
    return haloIndex_f.end();
  }

  // get the color of halo i
  Float3 getColor(int i)
  {
    return thrust::make_tuple(haloColorsR[i], haloColorsG[i], haloColorsB[i]);
  }

	// return haloIndex vector
  thrust::device_vector<int> getHalos()
	{
  	return haloIndex;
	}

  // get unique halo ids & numOfHalos
  void getUniqueHalos(int particleSize)
  {
    thrust::device_vector<int> haloUnique(numOfParticles);
    thrust::copy(haloIndex.begin(), haloIndex.end(), haloUnique.begin());
    thrust::sort(haloUnique.begin(), haloUnique.end()); //sort halo ids

    thrust::device_vector<int> haloSize(numOfParticles);
    thrust::fill(haloSize.begin(), haloSize.end(), 1);

    thrust::device_vector<int> a(numOfParticles);
    thrust::device_vector<int> b(numOfParticles);

    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end; 
    new_end = thrust::reduce_by_key(haloUnique.begin(), haloUnique.end(), haloSize.begin(), a.begin(), b.begin());

		//get number of halos
    int numUniqueHalos = thrust::get<0>(new_end) - a.begin();

    // get the number of invalid halos & their ids
    thrust::device_vector<int>::iterator new_end1 = thrust::remove_if(a.begin(), thrust::get<0>(new_end), b.begin(), validHalo(particleSize));
    int numOfInvalidHalos = new_end1 - a.begin();

    // for all particals in invalid halos, set halo id to -1
		for(int i=0; i<numOfInvalidHalos; i++)
			thrust::replace(haloIndex.begin(), haloIndex.end(), ((int)a[i]), -1);

		// get the number of valid halos & their ids
		new_end  = thrust::reduce_by_key(haloUnique.begin(), haloUnique.end(), haloSize.begin(), a.begin(), b.begin());
		new_end1 = thrust::remove_if(a.begin(), thrust::get<0>(new_end), b.begin(), invalidHalo(particleSize));

		numOfHalos = new_end1 - a.begin();

    haloIndexUnique = thrust::device_vector<int>(numOfHalos);
    thrust::copy(a.begin(), a.begin()+numOfHalos, haloIndexUnique.begin());

    haloUnique.clear(); haloSize.clear(); a.clear(); b.clear();
  }

  // get the halo ids from the input particle ids
  void getHaloIdFromInput()
  {
    thrust::device_vector<int> tmp; tmp.resize(numOfParticles);
    thrust::fill(tmp.begin(), tmp.end(), -1);
    for(int i=0; i<numOfHalos; i++)
    {
      int ind = haloIndexUnique[i];   int indNew = -1;
      for(int j=0; j<numOfParticles; j++)
      {
        if(haloIndex[j]==ind)
          indNew = (indNew==-1) ? leafI[j] : ((leafI[j]<indNew)?leafI[j]:indNew);
      }

      thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
          checkId(thrust::raw_pointer_cast(&*haloIndex.begin()),
                  thrust::raw_pointer_cast(&*tmp.begin()),
                  ind, indNew));
    }
    thrust::copy(tmp.begin(), tmp.end(), haloIndex.begin());
  }

  // check whether if of a certain item in haloIndex equals ind given
  struct checkId
  {
    int *haloIndex, *tmp;
    int ind, indNew;

    __host__ __device__
    checkId(int *haloIndex, int *tmp, int ind, int indNew) :
      haloIndex(haloIndex), tmp(tmp), ind(ind), indNew(indNew) {}

    __host__ __device__
    void operator()(int i)
    {
      if(haloIndex[i]==ind)
        tmp[i] = indNew;
    }
  };

  // check whether number of particles in this halo exceed particleSize
  struct validHalo
  {
    int particleSize;

		__host__ __device__
    validHalo(int particleSize) : particleSize(particleSize) {}

    __host__ __device__
    bool operator()(int i) { return i >= particleSize; }
  };

	// check whether number of particles in this halo DOES NOT exceed particleSize
	struct invalidHalo
  {
    int particleSize;

		__host__ __device__
    invalidHalo(int particleSize) : particleSize(particleSize) {}

    __host__ __device__
    bool operator()(int i) { return i < particleSize; }
  };
};

}

#endif /* HALO_H_ */

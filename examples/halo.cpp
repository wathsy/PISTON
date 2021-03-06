
using namespace std;

//#include <piston/halo_naive.h>
//#include <piston/halo_kd.h>
//#include <piston/halo_vtk.h>
#include <piston/halo_merge.h>
//#include <piston/halo_cc.h>

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

using namespace piston;

// given three vectors, compare the i the  element of the first two vectors& write 0 or 1 in the third vector
struct compare
{
	int *a, *b, *c;

	__host__ __device__
	compare(int* a, int* b, int* c) :
		a(a), b(b), c(c) {}

	__host__ __device__
	void operator()(int i)
	{
		if(a[i] != b[i])
    {
      c[i] = 1;
      //std::cout << i << " " << a[i] << " " << b[i] << std::endl;
    }
	}
};

// given two vectors, compare their elements
void compareResults(thrust::device_vector<int> a, thrust::device_vector<int> b, int numOfParticles, string txt)
{
	thrust::device_vector<int> c(numOfParticles);
	thrust::fill(c.begin(), c.end(), 0);

	thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
			compare(thrust::raw_pointer_cast(&*a.begin()), thrust::raw_pointer_cast(&*b.begin()), thrust::raw_pointer_cast(&*c.begin())));
	int count = thrust::reduce(c.begin(), c.begin() + numOfParticles);

	std::string output = (count==0) ? txt+" - Result is the same" : txt+" - Result is NOT the same";
  std::cout << output << std::endl;
	if(count != 0) std::cout << "count " << count << std::endl << std::endl;
}

// given one vector & a txt file with results, compare their elements
void compareResultsTxt(string filename, int numOfParticles, thrust::device_vector<int> d, string txt)
{
	int num = 0;
	std::string line;
	thrust::device_vector<int> items(numOfParticles);

	std::ifstream *myfile = new std::ifstream(filename.c_str(), std::ios::in);
	while(!myfile->eof())
	{
		getline(*myfile,line);

		if(line=="") continue;

		if(num<numOfParticles)
			items[num++] = atof(strtok((char*)line.c_str(), " "));

		for(int i=1; i<15; i++)
		{
			if(num<numOfParticles)
				items[num++] = atof(strtok(NULL, " "));
		}
	}

	int count = 0;
	for(int i=0; i<numOfParticles; i++)
	{
		if(d[i] != items[i]) 
		{
			//std::cout << i << " " << d[i] << " " << items[i] << std::endl;
			count++;
		}
	}

	std::string output = (count==0) ? txt+" - Result is the same" : txt+" - Result is NOT the same";
	std::cout << output << std::endl << std::endl;
	if(count != 0) std::cout << "count " << count << std::endl << std::endl;
}

// given one vector & a ascii file with results, compare their elements
void compareResultsAscii(string filename, int numOfParticles, thrust::device_vector<int> d, string txt)
{
	std::string line;
	thrust::device_vector<int> ids(numOfParticles);
	thrust::device_vector<int> items(numOfParticles);

	std::ifstream *myfile = new std::ifstream(filename.c_str(), std::ios::in);

	int i=0;
	while(!myfile->eof() && i++<7)
		getline(*myfile,line);		

	i = 0;
	while(!myfile->eof())
	{
		getline(*myfile,line);		

		if(line=="") continue;
	
		float x = atof(strtok((char*)line.c_str(), "\t"));
		float y = atof(strtok(NULL, "\t"));
		float z = atof(strtok(NULL, "\t"));

		float vx = atof(strtok(NULL, "\t"));
		float vy = atof(strtok(NULL, "\t"));
		float vz = atof(strtok(NULL, "\t"));
	
		int id = atoi(strtok(NULL, "\t"));
		int hid = atoi(strtok(NULL, "\t"));

		ids[i]   = id;
		items[i] = hid;
		i++;
	}

	int count = 0;
	for(int i=0; i<numOfParticles; i++)
	{
		if(d[i]!=items[i]) count++;
	}

	std::string output = (count==0) ? txt +" - Result is the same" : txt +" - Result is NOT the same";
	std::cout << output << std::endl << std::endl;
	if(count != 0) std::cout << "count " << count << std::endl << std::endl;
}


int main(int argc, char* argv[])
{
	if (argc < 11)
	{
		std::cout << "Usage:" << std::endl;
		std::cout << "haloGPU filename format min_ll max_ll l_length p_size np rL n k " << std::endl;
		std::cout << "OR" << std::endl;
		std::cout << "haloOMP filename format min_ll max_ll l_length p_size np rL n k\n" << std::endl;
		return 1;
	}

//  {
//    // serially generate 1M random numbers on the host
//    thrust::host_vector<int> h_vec(1 << 20);
//    thrust::sequence(h_vec.begin(), h_vec.end());
//
//    // transfer data to OpenMP
//    thrust::omp::vector<int> d_vec = h_vec;
//
//    // sort data in parallel with OpenMP
//    thrust::sort(d_vec.begin(), d_vec.end());
//
//    // transfer data back to host
//    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
//
//    // report the largest number
//    std::cout << "Largest number is " << h_vec.back() << std::endl;
//  }

  //---------------------------- set parameters

  halo *halo;

  char filename[1024]; // set file name
  sprintf(filename, "%s/%s", STRINGIZE_VALUE_OF(DATA_DIRECTORY), argv[1]);
  std::string format = argv[2];

  float min_linkLength = atof(argv[3]);
  float max_linkLength = atof(argv[4]);
  float linkLength     = atof(argv[5]);
	int   particleSize   = atof(argv[6]);

	// np - 128 & rL -150 for .hcosmo file
	// np - 256 & rL -64  for .cosmo file
  int   np = atoi(argv[7]);  // number of particles in one dimension
  float rL = atof(argv[8]);  // box length at a side
  int   n  = atoi(argv[9]);  // if you want a fraction of the file to load, use this.. 1/n
	int   k  = atoi(argv[10]); // k-way merge for global step in dendogram based halo finder

	std::cout << "min_linkLength " << min_linkLength << std::endl;
  std::cout << "max_linkLength " << max_linkLength << std::endl;
  std::cout << "linkLength " << linkLength << std::endl;
  std::cout << "particleSize " << particleSize << std::endl;
  std::cout << filename << std::endl;
  std::cout << std::endl;

  //---------------------------- run different versions

//  std::cout << "Naive result" << std::endl;
//
//  halo = new halo_naive(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> a = halo->getHalos();
//
//  std::cout << "VTK based result (thrust version)" << std::endl;/
//
//  halo = new halo_vtk(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> b = halo->getHalos();
//
//  std::cout << "Kdtree based result" << std::endl;
//
//  halo = new halo_kd(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> c = halo->getHalos();

  std::cout << "Merge tree based result" << std::endl;

  halo = new halo_merge(min_linkLength, max_linkLength, k, filename, format, n, np, rL);
  (*halo)(linkLength, particleSize);
  thrust::device_vector<int> d = halo->getHalos();

//  std::cout << "Sparse connected components result" << std::endl;
//
//  halo = new halo_cc(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> e = halo->getHalos();

  //---------------------------- compare results

//		std::cout << "Comparing results" << std::endl;

//  compareResultsAscii("/home/wathsy/Cosmo/PISTONSampleData/Small/output/m000.499.allparticles.ascii", halo->numOfParticles, d, "TestCase vs Mergetree");

//  compareResultsTxt((string)filename+"_Vtk.txt", halo->numOfParticles, d, "Vtk vs Mergetree");

//	compareResults(a, c, halo->numOfParticles, "Naive vs Kdtree");
//	compareResults(b, c, halo->numOfParticles, "Vtk (thrust version) vs Kdtree");
//	compareResults(c, d, halo->numOfParticles, "Kdtree vs Mergetree");
//	compareResults(d, e, halo->numOfParticles, "Mergetree vs Connected components");

//  std::cout << "a "; thrust::copy(a.begin(), a.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//  std::cout << "b "; thrust::copy(b.begin(), b.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//	std::cout << "c "; thrust::copy(c.begin(), c.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//	std::cout << "d "; thrust::copy(d.begin(), d.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;

	std::cout << std::endl;
  std::cout << "-----------------------------" << std::endl << std::endl;

	return 0;
}



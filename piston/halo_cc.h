#ifndef HALO_CC_H_
#define HALO_CC_H_

#include <piston/halo.h>
#include <piston/piston_math.h>

#include <cmath>

//#define SAMPLE_INPUT
//#define METHOD_NSQ
#define METHOD_PARTITION
#define PARTITION_FACTOR 3
//#define ROOT_GRAFT

namespace piston
{

class halo_cc : public halo
{
  int n;
  float threshold;
  thrust::device_vector<int> D;  thrust::device_vector<bool> S;  thrust::device_vector<int> R;  thrust::device_vector<int> M;  thrust::device_vector<int> I;
  thrust::device_vector<int> A;  thrust::device_vector<int> C;  thrust::device_vector<int> B;  thrust::device_vector<unsigned int> F;  thrust::device_vector<int> K;

public:

  halo_cc(std::string filename="", std::string format=".cosmo", int n_ = 1, int np=1, float rL=-1) : halo(filename, format, n_, np, rL)
  {
    n = numOfParticles;

    #ifdef SAMPLE_INPUT
        n = 13;
        leafX[0]  = 6;    leafY[0]  = 0;    leafZ[0]  = 0;
        leafX[1]  = 2;    leafY[1]  = 0;    leafZ[1]  = 0;
        leafX[2]  = 10;   leafY[2]  = 0;    leafZ[2]  = 0;
        leafX[3]  = 4;    leafY[3]  = 0;    leafZ[3]  = 0;
        leafX[4]  = 5;    leafY[4]  = 0;    leafZ[4]  = 0;
        leafX[5]  = 6;    leafY[5]  = -0.5; leafZ[5]  = 0;
        leafX[6]  = 10;   leafY[6]  = 1;    leafZ[6]  = 0;
        leafX[7]  = 1;    leafY[7]  = 1;    leafZ[7]  = 0;
        leafX[8]  = 1;    leafY[8]  = 0;    leafZ[8]  = 0;
        leafX[9]  = 6;    leafY[9]  = -1.5; leafZ[9]  = 0;
        leafX[10] = 2;    leafY[10] = -1;   leafZ[10] = 0;
        leafX[11] = 0;    leafY[11] = 0;    leafZ[11] = 0;
        leafX[12] = 3;    leafY[12] = 0;    leafZ[12] = 0;
        for (unsigned int i=0; i<n; i++) leafI[i] = 2*i;
    #endif

    // Initialize vectors
    D.resize(2*n);  S.resize(2*n);  R.resize(2*n);  M.resize(n);  I.resize(n);

    #ifdef METHOD_PARTITION
    B.resize(n);  A.resize(n);  C.resize(PARTITION_FACTOR*PARTITION_FACTOR*PARTITION_FACTOR*n);  F.resize(n);  K.resize(n);
    #endif
  }

  void operator()(float linkLength, int particleSize)
  {
    clear();

    threshold = linkLength*xscal;
    #ifdef SAMPLE_INPUT
        threshold = 1.25f;
    #endif
    std::cout << "Sizes: " << leafX.size() << " " << leafY.size() << " " << leafZ.size() << " " << haloIndex.size() << " " << n << std::endl;
    bool debug = false;

    // If debugging, output all edges
    if (debug)
    {
      for (unsigned int i=0; i<n; i++)
      {
        for (unsigned int j=i+1; j<n; j++)
        {
          float3 pi = make_float3(leafX[i], leafY[i], leafZ[i]);  float3 pj = make_float3(leafX[j], leafY[j], leafZ[j]);
          if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < threshold*threshold)
            std::cout << "Edge (" << i << ", " << j << ")" << std::endl;
        }
      }
    }

    struct timeval begin, mid1, mid2, end, diff1, diff2, diff3;
    gettimeofday(&begin, 0);

    #ifdef METHOD_PARTITION
        float minx = *(thrust::min_element(leafX.begin(), leafX.begin()+n));  float maxx = *(thrust::max_element(leafX.begin(), leafX.begin()+n));
        float miny = *(thrust::min_element(leafY.begin(), leafY.begin()+n));  float maxy = *(thrust::max_element(leafY.begin(), leafY.begin()+n));
        float minz = *(thrust::min_element(leafZ.begin(), leafZ.begin()+n));  float maxz = *(thrust::max_element(leafZ.begin(), leafZ.begin()+n));
        int num_bins_x = ceil((maxx-minx)/threshold);  int num_bins_y = ceil((maxy-minz)/threshold);  int num_bins_z = ceil((maxy-minz)/threshold);
        int max_bins = PARTITION_FACTOR*pow(n, 0.33);
        std::cout << "Original num bins: " << num_bins_x << " " << num_bins_y << " " << num_bins_z << " " << max_bins << std::endl;
        num_bins_x = std::min(max_bins, num_bins_x);  num_bins_y = std::min(max_bins, num_bins_y);  num_bins_z = std::min(max_bins, num_bins_z);
        int total_bins = num_bins_x*num_bins_y*num_bins_z;

        // Compute which bin each particle is in
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
            compute_bins(minx, maxx, miny, maxy, minz, maxz, num_bins_x, num_bins_y, num_bins_z, thrust::raw_pointer_cast(&*B.begin()),
                thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()), thrust::raw_pointer_cast(&*leafZ.begin())));
        if (debug) { std::cout << "Bins: " << std::endl; }
        if (debug) { thrust::copy(B.begin(), B.end(), std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

        // Sort the particles by bin
        thrust::sequence(A.begin(), A.end(), 0);
        thrust::sort_by_key(B.begin(), B.end(), thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin(), leafI.begin(), A.begin())));

        // Compute how many particles are in each bin, and then get the offsets to the start of each bin
        thrust::inclusive_scan_by_key(B.begin(), B.end(), thrust::make_constant_iterator(1), R.begin(), thrust::equal_to<int>(), thrust::plus<int>());
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
            get_bin_counts(n, thrust::raw_pointer_cast(&*R.begin()), thrust::raw_pointer_cast(&*B.begin()), thrust::raw_pointer_cast(&*C.begin())));
        thrust::exclusive_scan(C.begin(), C.begin()+total_bins, C.begin());

        // Mark active neighbor bins
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
            mark_active_neighbor_bins(n, total_bins, num_bins_x, num_bins_y, num_bins_z, threshold, thrust::raw_pointer_cast(&*F.begin()),
                thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()),
                thrust::raw_pointer_cast(&*leafZ.begin()), thrust::raw_pointer_cast(&*B.begin()), thrust::raw_pointer_cast(&*C.begin()), thrust::raw_pointer_cast(&*K.begin())));

        // Order array within bin by total number of comparisons to be performed
        thrust::sort_by_key(K.begin(), K.end(), thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin(), leafI.begin(), A.begin(), B.begin(), F.begin())));
        thrust::stable_sort_by_key(B.begin(), B.end(), thrust::make_zip_iterator(thrust::make_tuple(leafX.begin(), leafY.begin(), leafZ.begin(), leafI.begin(), A.begin(), F.begin())));

        // Initialize halo ids
        thrust::sequence(D.begin(), D.begin()+n, 0);
        thrust::sequence(D.begin()+n, D.begin()+2*n, 0);
    #endif

    gettimeofday(&mid1, 0);

    // Iterate until all vertices are in rooted stars
    int iter = 0;
    while (true)
    {
      std::cout << "Iteration " << iter++ << " " << threshold << " " << particleSize << std::endl;

      // Step 1: Graft trees onto smaller vertices of other trees
      #ifdef METHOD_NSQ
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n*n,
            graft(n, threshold, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()), thrust::raw_pointer_cast(&*leafZ.begin())));
      #else
      #ifdef METHOD_PARTITION
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
            graft_p(n, total_bins, num_bins_x, num_bins_y, num_bins_z, threshold, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()),
                thrust::raw_pointer_cast(&*leafZ.begin()), thrust::raw_pointer_cast(&*B.begin()), thrust::raw_pointer_cast(&*C.begin()), thrust::raw_pointer_cast(&*F.begin())));
      #else
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
            graft_n(n, threshold, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()), thrust::raw_pointer_cast(&*leafZ.begin())));
      #endif
      #endif
      if (debug) { thrust::copy(D.begin(), D.end(), std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

      #ifdef ROOT_GRAFT
        // Step 2a: Compute which vertices are part of a star
        thrust::fill(S.begin(), S.end(), true);
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+2*n,
            is_star1(thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*S.begin())));
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+2*n,
            is_star2(thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*S.begin())));
        if (debug) { thrust::copy(S.begin(), S.end(), std::ostream_iterator<bool>(std::cout, " "));   std::cout << std::endl << std::endl; }

        // Step 2b: Copy parent pointers to avoid race conditions in Step 2c
        thrust::copy(D.begin(), D.end(), R.begin());
        if (debug) { thrust::copy(R.begin(), R.end(), std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

        // Step 2c: Graft rooted stars onto other trees if possible
        #ifdef METHOD_NSQ
          thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n*n,
              root_graft(n, threshold, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*S.begin()), thrust::raw_pointer_cast(&*R.begin()),
                  thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()), thrust::raw_pointer_cast(&*leafZ.begin())));
        #else
        #ifdef METHOD_PARTITION
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
            root_graft_p(n, total_bins, num_bins_x, num_bins_y, num_bins_z, threshold, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*S.begin()), thrust::raw_pointer_cast(&*R.begin()),
                thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()), thrust::raw_pointer_cast(&*leafZ.begin()),
                thrust::raw_pointer_cast(&*B.begin()), thrust::raw_pointer_cast(&*C.begin()), thrust::raw_pointer_cast(&*F.begin())));
        #else
          thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
              root_graft_n(n, threshold, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*S.begin()), thrust::raw_pointer_cast(&*R.begin()),
                  thrust::raw_pointer_cast(&*leafX.begin()), thrust::raw_pointer_cast(&*leafY.begin()), thrust::raw_pointer_cast(&*leafZ.begin())));
        #endif
        #endif
        if (debug) { thrust::copy(D.begin(), D.end(), std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }
      #endif

      // Step 3a: Compute which vertices are part of a rooted star
      thrust::fill(S.begin(), S.end(), true);
      thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+2*n,
          is_star1(thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*S.begin())));
      thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+2*n,
          is_star2(thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*S.begin())));
      if (debug) { thrust::copy(S.begin(), S.end(), std::ostream_iterator<bool>(std::cout, " "));   std::cout << std::endl << std::endl; }

      // Step 3b: If all vertices are in rooted stars, algorithm is complete
      bool allStars = thrust::reduce(S.begin(), S.end(), true, thrust::logical_and<bool>());
      if (allStars) { std::cout << "All rooted stars; algorithm complete" << std::endl; break; }

      // Step 3c: Otherwise, do pointer jumping
      else
      {
        thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+2*n, pointer_jump(thrust::raw_pointer_cast(&*D.begin())));
        if (debug) { thrust::copy(D.begin(), D.end(), std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }
      }
    }

    // Output computed halo ids for each vertex
    if (debug) { std::cout << "Final halo ids: " << std::endl; }
    if (debug) { thrust::copy(D.begin(), D.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

    gettimeofday(&mid2, 0);

    // Get particle indexes from input
    thrust::transform(D.begin(), D.begin()+n, D.begin(), id_from_input(thrust::raw_pointer_cast(&*leafI.begin())));
    if (debug) { std::cout << "Input lookup result: " << std::endl; }
    if (debug) { thrust::copy(D.begin(), D.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

    // Do a sort by key on particle ids and then on halo id to get particle ids in increasing order grouped by halo id
    thrust::sequence(M.begin(), M.end(), 0);
    thrust::copy(M.begin(), M.end(), I.begin());
    thrust::transform(M.begin(), M.begin()+n, M.begin(), id_from_input(thrust::raw_pointer_cast(&*leafI.begin())));
    thrust::sort_by_key(M.begin(), M.begin()+n, thrust::make_zip_iterator(thrust::make_tuple(D.begin(), I.begin())));
    thrust::stable_sort_by_key(D.begin(), D.begin()+n, thrust::make_zip_iterator(thrust::make_tuple(M.begin(), I.begin())));
    if (debug) { std::cout << "Sort result: " << std::endl; }
    if (debug) { thrust::copy(D.begin(), D.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }
    if (debug) { thrust::copy(M.begin(), M.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

    // Do a segmented min-scan to get the minimum particle id in each halo
    thrust::inclusive_scan_by_key(D.begin(), D.begin()+n, M.begin(), D.begin(), thrust::equal_to<int>(), thrust::minimum<int>());
    thrust::fill(D.begin()+n, D.end(), -1);
    if (debug) { std::cout << "Scan result: " << std::endl; }
    if (debug) { thrust::copy(D.begin(), D.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

    // Determine the number of particles in each halo
    thrust::inclusive_scan_by_key(D.begin(), D.end(), thrust::make_constant_iterator(1), R.begin(), thrust::equal_to<int>(), thrust::plus<int>());
    thrust::inclusive_scan_by_key(D.rbegin(), D.rend(), R.rbegin(), R.rbegin(), thrust::equal_to<int>(), thrust::maximum<int>());
    if (debug) { std::cout << "Count result: " << std::endl; }
    if (debug) { thrust::copy(R.begin(), R.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

    // Count the number of valid halos
    numOfHalos = thrust::transform_reduce(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n,
        valid_halo(particleSize, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*R.begin())), 0, thrust::plus<int>());
    if (debug) std::cout << "Number of halos: " << numOfHalos << std::endl;

    // Remove invalid halos
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(0)+n, D.begin(), remove_invalid_halos(particleSize, thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*R.begin())));
    if (debug) { std::cout << "Remove invalid result: " << std::endl; }
    if (debug) { thrust::copy(D.begin(), D.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

    // Do another sort by key to reorder minimum halo ids back in sequential order of particle ids
    thrust::sort_by_key(I.begin(), I.begin()+n, D.begin());
    if (debug) { std::cout << "Minimum halo ids: " << std::endl; }
    if (debug) { thrust::copy(D.begin(), D.begin()+n, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl; }

    // Set haloIndex vector
    thrust::copy(D.begin(), D.begin()+n, haloIndex.begin());
    #ifdef METHOD_PARTITION
        thrust::sort_by_key(A.begin(), A.end(), haloIndex.begin());
    #endif

    gettimeofday(&end, 0);

    // Report timings
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed: " << seconds1 << " s for initialization"<< std::endl;
    timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed: " << seconds2 << " s for iterations"<< std::endl;
    timersub(&end, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed: " << seconds3 << " s for post-processing"<< std::endl;

    std::cout << "Number of Particles : " << numOfParticles <<  " Number of Halos found : " << numOfHalos << std::endl << std::endl;
  }


  struct graft : public thrust::unary_function<int, void>
  {
    int n;
    float t;
    int* D;
    float *X, *Y, *Z;

    graft(int n, float t, int* D, float* X, float* Y, float* Z) : n(n), t(t*t), D(D), X(X), Y(Y), Z(Z) {};

    __host__ __device__
    void operator() (int edge_id)
    {
      int i = edge_id / n;
      int j = edge_id % n;

      float3 pi = make_float3(X[i], Y[i], Z[i]);  float3 pj = make_float3(X[j], Y[j], Z[j]);
      if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < t)
      {
        if ((D[i] == D[D[i]]) && (D[j] < D[i]))
          D[D[i]] = D[j];
      }
    }
  };


  struct graft_n : public thrust::unary_function<int, void>
  {
    int n;
    float t;
    int* D;
    float *X, *Y, *Z;

    graft_n(int n, float t, int* D, float* X, float* Y, float* Z) : n(n), t(t*t), D(D), X(X), Y(Y), Z(Z) {};

    __host__ __device__
    void operator() (int i)
    {
      float3 pi = make_float3(X[i], Y[i], Z[i]);
      for (unsigned int j=0; j<n; j++)
      {
        if (i == j) continue;
        float3 pj = make_float3(X[j], Y[j], Z[j]);
        if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < t)
        {
          if ((D[i] == D[D[i]]) && (D[j] < D[i]))
            D[D[i]] = D[j];
        }
      }
    }
  };


  struct graft_p : public thrust::unary_function<int, void>
  {
    int n, tb, nbx, nby, nbz;
    float t;
    int* D;
    float *X, *Y, *Z;
    int* B;
    int* C;
    unsigned int* F;

    graft_p(int n, int tb, int nbx, int nby, int nbz, float t, int* D, float* X, float* Y, float* Z, int* B, int* C, unsigned int* F) :
      n(n), tb(tb), nbx(nbx), nby(nby), nbz(nbz), t(t*t), D(D), X(X), Y(Y), Z(Z), B(B), C(C), F(F) {};

    __host__ __device__
    void operator() (int i)
    {
      int bin = B[i];
      const int xbin = bin % nbx;
      const int ybin = (bin / nbx) % nby;
      const int zbin = bin / (nbx*nby);
      float3 pi = make_float3(X[i], Y[i], Z[i]);

      unsigned int flag = F[i];
      for (int x=xbin-1; x<=xbin+1; x++)
      {
        for (int y=ybin-1; y<=ybin+1; y++)
        {
          for (int z=zbin-1; z<=zbin+1; z++)
          {
            if (flag & 0x1)
            {
              int cur_bin = x + y*nbx + z*nbx*nby;
              int start_index = C[cur_bin];
              int end_index = n;  if (cur_bin+1 < tb) end_index = C[cur_bin+1];
              for (unsigned int j=start_index; j<end_index; j++)
              {
                float3 pj = make_float3(X[j], Y[j], Z[j]);
                if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < t)
                {
                  if ((D[i] == D[D[i]]) && (D[j] < D[i]))
                    D[D[i]] = D[j];
                }
              }
            }
            flag = flag >> 1;
          }
        }
      }
    }
  };


  struct is_star1 : public thrust::unary_function<int, void>
  {
    int* D;
    bool* S;

    is_star1(int* D, bool* S) : D(D), S(S) {};

    __host__ __device__
    void operator() (int id)
    {
      if (D[id] != D[D[id]]) { S[id] = S[D[id]] = S[D[D[id]]] = false; }
    }
  };


  struct is_star2 : public thrust::unary_function<int, void>
  {
    int* D;
    bool* S;

    is_star2(int* D, bool* S) : D(D), S(S) {};

    __host__ __device__
    void operator() (int id)
    {
      S[id] = S[D[id]];
    }
  };


  struct root_graft : public thrust::unary_function<int, void>
  {
    int n;
    float t;
    int* D;
    bool* S;
    int* D2;
    float *X, *Y, *Z;

    root_graft(int n, float t, int* D, bool* S, int* D2, float* X, float* Y, float* Z) : n(n), t(t*t), D(D), S(S), D2(D2), X(X), Y(Y), Z(Z) {};

    __host__ __device__
    void operator() (int edge_id)
    {
      int i = edge_id / n;
      int j = edge_id % n;

      float3 pi = make_float3(X[i], Y[i], Z[i]);  float3 pj = make_float3(X[j], Y[j], Z[j]);
      if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < t)
      {
        if ((S[i]) && (D[i] < D[j]))
          D[D2[i]] = D[j];
      }
    }
  };


  struct root_graft_n : public thrust::unary_function<int, void>
  {
    int n;
    float t;
    int* D;
    bool* S;
    int* D2;
    float *X, *Y, *Z;

    root_graft_n(int n, float t, int* D, bool* S, int* D2, float* X, float* Y, float* Z) : n(n), t(t*t), D(D), S(S), D2(D2), X(X), Y(Y), Z(Z) {};

    __host__ __device__
    void operator() (int edge_id)
    {
      int i = edge_id;
      float3 pi = make_float3(X[i], Y[i], Z[i]);
      for (unsigned int j=0; j<n; j++)
      {
        if (i == j) continue;
        float3 pj = make_float3(X[j], Y[j], Z[j]);
        if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < t)
        {
          if ((S[i]) && (D[i] < D[j]))
            D[D2[i]] = D[j];
        }
      }
    }
  };


  struct root_graft_p : public thrust::unary_function<int, void>
  {
    int n, tb, nbx, nby, nbz;
    float t;
    int* D;
    bool* S;
    int* D2;
    float *X, *Y, *Z;
    int* B;
    int* C;
    unsigned int* F;

    root_graft_p(int n, int tb, int nbx, int nby, int nbz, float t, int* D, bool* S, int* D2, float* X, float* Y, float* Z, int* B, int* C, unsigned int* F) :
      n(n), tb(tb), nbx(nbx), nby(nby), nbz(nbz), t(t*t), D(D), S(S), D2(D2), X(X), Y(Y), Z(Z), B(B), C(C), F(F) {};

    __host__ __device__
    void operator() (int i)
    {
      int bin = B[i];
      const int xbin = bin % nbx;
      const int ybin = (bin / nbx) % nby;
      const int zbin = bin / (nbx*nby);
      float3 pi = make_float3(X[i], Y[i], Z[i]);

      unsigned int flag = F[i];
      for (int x=xbin-1; x<=xbin+1; x++)
      {
        for (int y=ybin-1; y<=ybin+1; y++)
        {
          for (int z=zbin-1; z<=zbin+1; z++)
          {
            if (flag & 0x1)
            {
              int cur_bin = x + y*nbx + z*nbx*nby;
              int start_index = C[cur_bin];
              int end_index = n;  if (cur_bin+1 < tb) end_index = C[cur_bin+1];
              for (unsigned int j=start_index; j<end_index; j++)
              {
                float3 pj = make_float3(X[j], Y[j], Z[j]);
                if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < t)
                {
                  if ((S[i]) && (D[i] < D[j]))
                    D[D2[i]] = D[j];
                }
              }
            }
            flag = flag >> 1;
          }
        }
      }
    }
  };


  struct pointer_jump : public thrust::unary_function<int, void>
  {
    int* D;

    pointer_jump(int* D) : D(D) {};

    __host__ __device__
    void operator() (int id)
    {
      D[id] = D[D[id]];
    }
  };


  struct id_from_input : public thrust::unary_function<int, int>
  {
    int* I;

    id_from_input(int* I) : I(I) {};

    __host__ __device__
    int operator() (int id)
    {
      return (I[id]);
    }
  };


  struct valid_halo : public thrust::unary_function<int, int>
  {
    int t;
    int* D;
    int* R;

    valid_halo(int t, int* D, int* R) : t(t), D(D), R(R) {};

    __host__ __device__
    int operator() (int id)
    {
      if (id == 0) return 1;
      if ((D[id] != D[id-1]) && (R[id] >= t)) return 1;
      return 0;
    }
  };


  struct remove_invalid_halos : public thrust::unary_function<int, int>
  {
    int t;
    int* D;
    int* R;

    remove_invalid_halos(int t, int* D, int* R) : t(t), D(D), R(R) {};

    __host__ __device__
    int operator() (int id)
    {
      if (R[id] < t) return -1;
      return (D[id]);
    }
  };


  struct compute_bins : public thrust::unary_function<int, void>
  {
    float minx, maxx, miny, maxy, minz, maxz;
    int nbx, nby, nbz;
    int* B;
    float *X, *Y, *Z;

    compute_bins(float minx, float maxx, float miny, float maxy, float minz, float maxz, int nbx, int nby, int nbz, int* B, float* X, float* Y, float* Z) :
      minx(minx), maxx(maxx), miny(miny), maxy(maxy), minz(minz), maxz(maxz), nbx(nbx), nby(nby), nbz(nbz), B(B), X(X), Y(Y), Z(Z) {};

    __host__ __device__
    void operator() (int id)
    {
      int xbin = 0;  if (nbx > 1) { xbin = nbx*(X[id]-minx)/(maxx-minx);  if (xbin >= nbx) xbin = nbx-1; }
      int ybin = 0;  if (nby > 1) { ybin = nby*(Y[id]-miny)/(maxy-miny);  if (ybin >= nby) ybin = nby-1; }
      int zbin = 0;  if (nbz > 1) { zbin = nbz*(Z[id]-minz)/(maxz-minz);  if (zbin >= nbz) zbin = nbz-1; }
      B[id] = xbin + ybin*nbx + zbin*nbx*nby;
    }
  };


  struct get_bin_counts : public thrust::unary_function<int, void>
  {
    int n;
    int* R;
    int* B;
    int* C;

    get_bin_counts(int n, int* R, int* B, int* C) : n(n), R(R), B(B), C(C) {};

    __host__ __device__
    void operator() (int id)
    {
      if ((id == n-1) || (B[id] != B[id+1])) C[B[id]] = R[id];
    }
  };


  struct mark_active_neighbor_bins : public thrust::unary_function<int, void>
  {
    int n, tb, nbx, nby, nbz;
    float t;
    unsigned int* F;
    float *X, *Y, *Z;
    int* B;
    int* C;
    int* K;

    mark_active_neighbor_bins(int n, int tb, int nbx, int nby, int nbz, float t, unsigned int* F, float* X, float* Y, float* Z, int* B, int* C, int* K) :
      n(n), tb(tb), nbx(nbx), nby(nby), nbz(nbz), t(t*t), F(F), X(X), Y(Y), Z(Z), B(B), C(C), K(K) {};

    __host__ __device__
    void operator() (int i)
    {
      int bin = B[i];
      const int xbin = bin % nbx;
      const int ybin = (bin / nbx) % nby;
      const int zbin = bin / (nbx*nby);
      float3 pi = make_float3(X[i], Y[i], Z[i]);

      unsigned int flag = 0;
      unsigned int bcnt = 1;
      int ncnt = 0;

      for (int x=xbin-1; x<=xbin+1; x++)
      {
        for (int y=ybin-1; y<=ybin+1; y++)
        {
          for (int z=zbin-1; z<=zbin+1; z++)
          {
            if ((x >= 0) && (x < nbx) && (y >= 0) && (y < nby) && (z >= 0) && (z < nbz))
            {
              int cur_bin = x + y*nbx + z*nbx*nby;
              int start_index = C[cur_bin];
              int end_index = n;  if (cur_bin+1 < tb) end_index = C[cur_bin+1];
              for (unsigned int j=start_index; j<end_index; j++)
              {
                float3 pj = make_float3(X[j], Y[j], Z[j]);
                if ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z) < t)
                {
                  flag = flag | bcnt;
                  ncnt += (end_index - start_index);
                  break;
                }
              }
            }
            bcnt = bcnt << 1;
          }
        }
      }
      F[i] = flag;
      K[i] = ncnt;
    }
  };
};

}

#endif

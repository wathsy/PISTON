/*
 * threshold_geometry.h
 *
 *  Created on: Sep 21, 2011
 *      Author: ollie
 */

#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>

#include <piston/image3d.h>
#include <piston/sphere.h>
#include <piston/cutil_math.h>

template <typename InputDataSet, typename ThresholdFunction>
struct threshold_geometry
{
    typedef typename InputDataSet::PointDataIterator InputPointDataIterator;
    typedef thrust::host_vector<int> IndicesContainer;
    typedef typename IndicesContainer::iterator IndicesIterator;
    typedef typename thrust::counting_iterator<int, thrust::host_space_tag>	CountingIterator;
    typedef thrust::host_vector<bool> ValidFlagsContainer;
    typedef typename ValidFlagsContainer::iterator ValidFlagsIterator;

    typedef typename thrust::host_vector<float4>::iterator VerticesIterator;

    InputDataSet &input;
    ThresholdFunction &threshold;

    threshold_geometry(InputDataSet input, ThresholdFunction threshold) :
	input(input), threshold(threshold) {}

    void operator()() {
	const int NCells = (input.xdim - 1)*(input.ydim - 1)*(input.zdim - 1);

	thrust::host_vector<bool> valid_cell_flags(NCells);
	thrust::transform(CountingIterator(0), CountingIterator(0)+NCells,
	                  valid_cell_flags.begin(),
	                  threshold_cell(input, threshold));

	thrust::host_vector<int> valid_cell_enum(NCells);
	// test and enumerate cells that pass threshold
//	thrust::transform_inclusive_scan(CountingIterator(0), CountingIterator(0)+NCells,
//	                                 valid_cell_enum.begin(),
//	                                 threshold_cell(input, threshold),
//	                                 thrust::plus<int>());
	// enumerate valid cells
	thrust::inclusive_scan(valid_cell_flags.begin(), valid_cell_flags.end(),
	                       valid_cell_enum.begin());
	int num_valid_cells = valid_cell_enum.back();

	std::cout << "valid cells enum: ";
	thrust::copy(valid_cell_enum.begin(), valid_cell_enum.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	std::cout << "number of valid cells: " << num_valid_cells << std::endl;

	thrust::host_vector<int> valid_cell_indices(num_valid_cells);
	// generate indices to cells that pass threshold
	thrust::upper_bound(valid_cell_enum.begin(), valid_cell_enum.end(),
	                    CountingIterator(0), CountingIterator(0)+num_valid_cells,
	                    valid_cell_indices.begin());
	std::cout << "indices to valid cells: ";
	thrust::copy(valid_cell_indices.begin(), valid_cell_indices.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	thrust::host_vector<int> num_boundary_cell_neighbors(num_valid_cells);
	thrust::transform(valid_cell_indices.begin(), valid_cell_indices.end(),
	                  num_boundary_cell_neighbors.begin(),
	                  boundary_cell_neighbors(input, valid_cell_flags.begin()));
	std::cout << "# of boundary cell neighbors: ";
	thrust::copy(num_boundary_cell_neighbors.begin(), num_boundary_cell_neighbors.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	thrust::host_vector<bool> boundary_cell_flags(num_valid_cells);
	thrust::transform(num_boundary_cell_neighbors.begin(), num_boundary_cell_neighbors.end(),
	                  boundary_cell_flags.begin(),
	                  is_boundary_cell());
	std::cout << "is boundary cell: ";
	thrust::copy(boundary_cell_flags.begin(), boundary_cell_flags.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	thrust::host_vector<int> boundary_cell_enum(num_valid_cells);
	thrust::inclusive_scan(boundary_cell_flags.begin(), boundary_cell_flags.end(),
	                       boundary_cell_enum.begin());
	std::cout << "boundary cells enum: ";
	thrust::copy(boundary_cell_enum.begin(), boundary_cell_enum.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	int num_boundary_cells = boundary_cell_enum.back();
	std::cout << "number of boundary cells: " << num_boundary_cells << std::endl;

	thrust::host_vector<int> boundary_cell_indices(num_boundary_cells);
	thrust::upper_bound(boundary_cell_enum.begin(), boundary_cell_enum.end(),
	                    CountingIterator(0), CountingIterator(0)+num_boundary_cells,
	                    boundary_cell_indices.begin());
	std::cout << "indices to boundary cells in valid cells: ";
	thrust::copy(boundary_cell_indices.begin(), boundary_cell_indices.end(), std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	std::cout << "indices to boundary cells in all cells: ";
	thrust::copy(thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.begin()),
	             thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.end()),
	             std::ostream_iterator<int>(std::cout, " "));
	std::cout << std::endl;

	thrust::host_vector<float4> vertices(num_valid_cells*24);

	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(boundary_cell_indices.begin(),
	                                                              thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.begin()))),
	                 thrust::make_zip_iterator(thrust::make_tuple(boundary_cell_indices.end(),
	                                                              thrust::make_permutation_iterator(valid_cell_indices.begin(), boundary_cell_indices.begin()))),
	                 generate_quads(input, thrust::raw_pointer_cast(&*vertices.begin())));
	thrust::for_each(vertices.begin(), vertices.end(), print_float4());

    }

    struct print_float4 : public thrust::unary_function<float4, void>
    {
	__host__ __device__
	void operator() (float4 p) {
	    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
	}
    };

    // FixME: the input data type should really be cells rather than cell_ids
    struct threshold_cell : public thrust::unary_function<int, bool>
    {
	// FixME: constant iterator and/or iterator to const problem.
	InputDataSet &input;
	const ThresholdFunction &threshold;

	__host__ __device__
	threshold_cell(InputDataSet &input, ThresholdFunction threshold) :
	    input(input), threshold(threshold) {}

	__host__ __device__
	bool operator() (int cell_id) const {
	    const int xdim = input.xdim;
	    const int ydim = input.ydim;
	    const int zdim = input.zdim;

	    const int cells_per_layer = (xdim - 1) * (ydim - 1);

	    const int x = cell_id % (xdim - 1);
	    const int y = (cell_id / (xdim - 1)) % (ydim -1);
	    const int z = cell_id / cells_per_layer;

	    // indices to the eight vertices of the voxel
	    const int i0 = x    + y*xdim + z * xdim * ydim;
	    const int i1 = i0   + 1;
	    const int i2 = i0   + 1	+ xdim;
	    const int i3 = i0   + xdim;

	    const int i4 = i0   + xdim * ydim;
	    const int i5 = i1   + xdim * ydim;
	    const int i6 = i2   + xdim * ydim;
	    const int i7 = i3   + xdim * ydim;

	    InputPointDataIterator point_data = input.point_data_begin();

	    // scalar values of the eight vertices
	    const float f0 = *(point_data + i0);
	    const float f1 = *(point_data + i1);
	    const float f2 = *(point_data + i2);
	    const float f3 = *(point_data + i3);
	    const float f4 = *(point_data + i4);
	    const float f5 = *(point_data + i5);
	    const float f6 = *(point_data + i6);
	    const float f7 = *(point_data + i7);

	    // a cell is considered passing the threshold if all of its vertices
	    // are passing the threshold.
	    bool valid = threshold(f0);
	    valid &= threshold(f1);
	    valid &= threshold(f2);
	    valid &= threshold(f3);
	    valid &= threshold(f4);
	    valid &= threshold(f5);
	    valid &= threshold(f6);
	    valid &= threshold(f7);

	    std::cout << "cell id: " << cell_id << ", valid: " << valid << std::endl;
	    return valid;
	}
    };

    struct boundary_cell_neighbors : public thrust::unary_function<int, int>
    {
	const InputDataSet &input;
	const ValidFlagsIterator &valid_cell_flags;

	__host__ __device__
	boundary_cell_neighbors(InputDataSet input, ValidFlagsIterator valid_cell_flags) :
	    input(input), valid_cell_flags(valid_cell_flags) {}

	__host__ __device__
	int operator() (int valid_cell_id) const {
	    const int xdim = input.xdim;
	    const int ydim = input.ydim;
	    const int zdim = input.zdim;
	    const int cells_per_layer = (xdim - 1) * (ydim - 1);

	    // cell ids of the cell's six neighbors
	    const int n0 = valid_cell_id - (xdim - 1);
	    const int n1 = valid_cell_id + 1;
	    const int n2 = valid_cell_id + (xdim - 1);
	    const int n3 = valid_cell_id - 1;
	    const int n4 = valid_cell_id - cells_per_layer;
	    const int n5 = valid_cell_id + cells_per_layer;

	    // if the cell is at the boundary of the whole data set,
	    // it has a boundary cell neighbor at that face.
	    const int x = valid_cell_id % (xdim - 1);
	    const int y = (valid_cell_id / (xdim - 1)) % (ydim -1);
	    const int z = valid_cell_id / cells_per_layer;

	    int boundary = *(valid_cell_flags + n0) || (y == 0);
	    boundary    += *(valid_cell_flags + n1) || (x == (xdim - 1));
	    boundary    += *(valid_cell_flags + n2) || (y == (ydim - 1));
	    boundary    += *(valid_cell_flags + n3) || (x == 0);
	    boundary    += *(valid_cell_flags + n4) || (z == 0);
	    boundary    += *(valid_cell_flags + n5) || (z == (zdim - 1));

	    return boundary;
	}
    };

    struct is_boundary_cell : public thrust::unary_function<int, bool>
    {
	__host__ __device__
	bool operator() (int num_boundary_cell_neighbors) const {
	    return num_boundary_cell_neighbors != 0;
	}
    };

    // FixME: the input data type should really be cells rather than cell_ids
    // FixME: should only generate quads for real outer/boundary faces
    struct generate_quads : public thrust::unary_function<thrust::tuple<int, int>, void>
    {
	InputDataSet &input;
	float4 *vertices;

	generate_quads(InputDataSet input, float4 *vertices) :
	    input(input), vertices(vertices) {}

	__host__ __device__
	void operator() (thrust::tuple<int, int> indices_tuple) {
	    const int valid_cell_id  = thrust::get<0>(indices_tuple);
	    const int global_cell_id = thrust::get<1>(indices_tuple);

	    std::cout << "valid cell id: " << valid_cell_id << ", global_cell_id: " << global_cell_id << std::endl;

	    const int vertices_for_faces[] =
	    {
		 0, 1, 5, 4, // face 0
		 1, 2, 6, 5, // face 1
		 2, 3, 7, 6,
		 0, 4, 7, 3,
		 0, 3, 2, 1,
		 4, 5, 6, 7
	    };
	    const int xdim = input.xdim;
	    const int ydim = input.ydim;
	    const int zdim = input.zdim;

	    const int cells_per_layer = (xdim - 1) * (ydim - 1);

	    const int x = global_cell_id % (xdim - 1);
	    const int y = (global_cell_id / (xdim - 1)) % (ydim -1);
	    const int z = global_cell_id / cells_per_layer;

	    // indices to the eight vertices of the voxel
	    const int i0 = x    + y*xdim + z * xdim * ydim;
	    const int i1 = i0   + 1;
	    const int i2 = i0   + 1	+ xdim;
	    const int i3 = i0   + xdim;

	    const int i4 = i0   + xdim * ydim;
	    const int i5 = i1   + xdim * ydim;
	    const int i6 = i2   + xdim * ydim;
	    const int i7 = i3   + xdim * ydim;

	    InputPointDataIterator point_data = input.point_data_begin();

	    // scalar values of the eight vertices
	    const float f0 = *(point_data + i0);
	    const float f1 = *(point_data + i1);
	    const float f2 = *(point_data + i2);
	    const float f3 = *(point_data + i3);
	    const float f4 = *(point_data + i4);
	    const float f5 = *(point_data + i5);
	    const float f6 = *(point_data + i6);
	    const float f7 = *(point_data + i7);

	    // position of the eight vertices
	    float3 p[8];
	    p[0] = make_float3(x, y, z);
	    p[1] = p[0] + make_float3(1.0f, 0.0f, 0.0f);
	    p[2] = p[0] + make_float3(1.0f, 1.0f, 0.0f);
	    p[3] = p[0] + make_float3(0.0f, 1.0f, 0.0f);
	    p[4] = p[0] + make_float3(0.0f, 0.0f, 1.0f);
	    p[5] = p[0] + make_float3(1.0f, 0.0f, 1.0f);
	    p[6] = p[0] + make_float3(1.0f, 1.0f, 1.0f);
	    p[7] = p[0] + make_float3(0.0f, 1.0f, 1.0f);

	    for (int v = 0; v < 24; v++) {
		*(vertices + valid_cell_id*24 + v) = make_float4(p[vertices_for_faces[v]]);
	    }
	}
    };

};

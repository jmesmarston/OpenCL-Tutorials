//a simple OpenCL kernel which adds two vectors A and B together into a third vector C
kernel void add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0); //Each launch gets a separate and unique ID which is obtained by get_global_id(0) function.
	C[id] = A[id] + B[id];
}

// Multiple the vectors together
kernel void mult(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	//C[id] = A[id] + B[id];
	C[id] = A[id] * B[id];
}

// mult and add vectors together in the form: C = A * B + B
kernel void mult_add(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	//C[id] = A[id] + B[id];
	C[id] = A[id] * B[id] + B[id];
}

//a simple smoothing kernel averaging values in a local window (radius 1)
kernel void avg_filter(global const int* A, global int* B) {
	int gid = get_global_id(0);
	int id = get_local_id(0);
	int local_size = get_local_size(0);
	printf("id = %d. gid = %d. local size = %d\n", id, gid, local_size);

	//for (int i = 0; i <= local_size; i++) {
		//printf("i = %d", i);
	if (id == 0) {
		printf("left bound con at i = % d\n", id);
		B[gid] = (A[local_size - 1] + A[id] + A[id + 1]) / 3;
	}
	else if (id == local_size-1) {
		printf("right bound con at i = % d\n", id);
		B[gid] = (A[id - 1] + A[id] + A[id - local_size]) / 3;
	}
	else {

		B[gid] = (A[id - 1] + A[id] + A[id + 1]) / 3;

	}
		//barrier(CLK_GLOBAL_MEM_FENCE);

	//}
	//B[id] = (A[id - 1] + A[id] + A[id + 1]) / 3;
	/*
	ans:
	A = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	C = [0, 1, 2, 3, 4, 5, 6, 7, 8, 5]
	Res:
	C = [3, 1, 2, 3, 4, 5, 6, 7, 8, 5]
	*/
}

//a simple 2D kernel
kernel void add2D(global const int* A, global const int* B, global int* C) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	int width = get_global_size(0);
	int height = get_global_size(1);
	int id = x + y * width;

	printf("id = %d x = %d y = %d w = %d h = %d\n", id, x, y, width, height);

	C[id] = A[id] + B[id];
}
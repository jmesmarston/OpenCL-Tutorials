//a simple OpenCL kernel which copies all pixels from A to B
kernel void identity(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	
	//printf("work item id = %d\n", id);

	if (id == 0) { // perform this part only once i.e. for work item 0
		//printf("work group size %d\n", get_local_size(0));
	}
	int loc_id = get_local_id(0);
	//printf("global id = %d, local id = %d\n", id, loc_id); // do it for each work item
	B[id] = A[id];
}

//a simple smoothing kernel averaging values in a local window (radius 1)
kernel void avg_filter(global const int* A, global int* B) {
	int gid = get_global_id(0);
	int id = get_local_id(0);
	int local_size = get_local_size(0);
	//printf("id = %d. gid = %d. local size = %d\n", id, gid, local_size);

	//for (int i = 0; i <= local_size; i++) {
		//printf("i = %d", i);
	if (id == 0) {
		//printf("left bound con at i = % d\n", id);
		B[gid] = (A[local_size - 1] + A[id] + A[id + 1]) / 3;
	}
	else if (id == local_size - 1) {
		//printf("right bound con at i = % d\n", id);
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

kernel void filter_r(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0)/3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	if (colour_channel == 0) {
		B[id] = A[id];
	}

	//this is just a copy operation, modify to filter out the individual colour channels
	//B[id] = A[id];
}


kernel void invert(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue

	B[id] = 255 -A[id];


	//this is just a copy operation, modify to filter out the individual colour channels
	//B[id] = A[id];
}


kernel void rgb2grey(global const uchar* A, global uchar* B) {
	int id = get_global_id(0);
	int image_size = get_global_size(0) / 3; //each image consists of 3 colour channels
	int colour_channel = id / image_size; // 0 - red, 1 - green, 2 - blue
	//printf("id = %d", id);

	/*From james:
	since colour_channel * 0 = 0 and it’s not used for red, colour_channel * 1 is colour_channel for green, 
	and colour_channel * 2 works for blue*/
	B[id] = A[id - (image_size * colour_channel)]; 


	//this is just a copy operation, modify to filter out the individual colour channels
	//B[id] = A[id];
}

//simple ND identity kernel
kernel void identityND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	B[id] = A[id];
}

//2D averaging filter
kernel void avg_filterND(global const uchar* A, global uchar* B) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	uint result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size];

		result /= 9;
	}

	B[id] = (uchar)result;
}

//2D 3x3 convolution kernel
kernel void convolutionND(global const uchar* A, global uchar* B, constant float* mask) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width*height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y*width + c*image_size; //global id in 1D space

	float result = 0;

	//simple boundary handling - just copy the original pixel
	if ((x == 0) || (x == width-1) || (y == 0) || (y == height-1)) {
		result = A[id];	
	} else {
		for (int i = (x-1); i <= (x+1); i++)
		for (int j = (y-1); j <= (y+1); j++) 
			result += A[i + j*width + c*image_size]*mask[i-(x-1) + j-(y-1)];
	}

	B[id] = (uchar)result;
}
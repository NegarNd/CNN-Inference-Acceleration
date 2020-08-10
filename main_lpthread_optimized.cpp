/********************************************
	MNIST classification based on Convolutional Neural Network
	Author: Seyed Ahmad Mirsalari
	University of Tehran
*********************************************/

// code is not allowed to modification
// except for Predict, forward and relu functions
// and marked part of main function

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h> 
#include "pthread.h"

#define LENGTH_KERNEL	5
#define LENGTH_FEATURE0	32
#define LENGTH_FEATURE1	28
#define LENGTH_FEATURE2	14
#define LENGTH_FEATURE3	10
#define	LENGTH_FEATURE4	5
#define LENGTH_FEATURE5	1

#define INPUT			1
#define LAYER1			6
#define LAYER2			6
#define LAYER3			16
#define LAYER4			16
#define LAYER5			120
#define LAYER6         256
#define OUTPUT          10
#define PADDING			2

#define FILE_TEST_IMAGE		"t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"t10k-labels-idx1-ubyte"

#define COUNT_TEST		10000

#define NUM_THREADS 3

typedef unsigned char uint8;
typedef uint8 image[28][28];

typedef struct Net
{
	double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];			//[1][6][5][5]
	double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];			//[6][16][5][5]
	double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];			//[16][120][5][5]
	double weight5_6[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][LAYER6];	//[120][256]
    double weight6_7[LAYER6][OUTPUT];	//[256][10]

	double bias0_1[LAYER1];//[6]
	double bias2_3[LAYER3];//[16]
	double bias4_5[LAYER5];//[120]
	double bias5_6[LAYER6];//[256]
	double bias6_7[OUTPUT];//[10]
}Net;

typedef struct Feature
{
	double input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];//32
	double layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];//28
	double layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];//14
	double layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];//10
	double layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];//5
	double layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];//1
	double layer6[LAYER6];//256
	double output[OUTPUT];//10
}Feature;

typedef struct input_param
{
	int offSet;
	int remainder;
	Net *netStructure;
	uint8 *testLabel;
	image *testData;
}input_param;

/*double relu(double x)
{
	// relu function body is allowed to modification
	if (x > 0)
		return x;
	return 0;
}*/

void forward(Net *net, Feature *features)
{
	// forward function body is allowed to modification
	//int i, j, x, y, o0, o1, w0, w1, l0, l1;
	double temp = 0;
	// convolution
	for (uint8 o0 = 0; o0 < 28; o0++)
		for (uint8 o1 = 0; o1 < 28; o1++)
		{	
			temp = features->layer1[0][o0][o1];

			temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][0][0][0] 
			+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][0][0][1]
			+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][0][0][2]
			+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][0][0][3]
			+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][0][0][4]
			                                                      
			+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][0][1][0]
			+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][0][1][1]
			+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][0][1][2]
			+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][0][1][3]
			+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][0][1][4]
			                                                        
			+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][0][2][0]
			+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][0][2][1]
			+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][0][2][2]
			+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][0][2][3]
			+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][0][2][4]
			                                                       
			+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][0][3][0]
			+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][0][3][1]
			+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][0][3][2]
			+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][0][3][3]
			+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][0][3][4]
			                                                        
			+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][0][4][0]
			+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][0][4][1]
			+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][0][4][2]
			+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][0][4][3]
			+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][0][4][4];
			
			//temp = relu(temp + net->bias0_1[0]);
			temp += net->bias0_1[0];
			if (temp < 0)
				temp = 0;

			features->layer1[0][o0][o1] = temp;
			
			
			
			temp = features->layer1[1][o0][o1];

			temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][1][0][0]
			+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][1][0][1]
			+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][1][0][2]
			+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][1][0][3]
			+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][1][0][4]
			                                                      
			+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][1][1][0]
			+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][1][1][1]
			+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][1][1][2]
			+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][1][1][3]
			+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][1][1][4]
			                                                        
			+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][1][2][0]
			+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][1][2][1]
			+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][1][2][2]
			+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][1][2][3]
			+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][1][2][4]
			                                                       
			+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][1][3][0]
			+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][1][3][1]
			+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][1][3][2]
			+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][1][3][3]
			+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][1][3][4]
			                                                       
			+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][1][4][0]
			+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][1][4][1]
			+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][1][4][2]
			+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][1][4][3]
			+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][1][4][4];
			
			//temp = relu(temp + net->bias0_1[1]);
			temp += net->bias0_1[1];
			if (temp < 0)
				temp = 0;
			
			features->layer1[1][o0][o1] = temp;
			
			
			
			temp = features->layer1[2][o0][o1];

			temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][2][0][0]
			+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][2][0][1]
			+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][2][0][2]
			+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][2][0][3]
			+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][2][0][4]
			                                                       
			+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][2][1][0]
			+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][2][1][1]
			+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][2][1][2]
			+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][2][1][3]
			+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][2][1][4]
			                                                       
			+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][2][2][0]
			+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][2][2][1]
			+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][2][2][2]
			+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][2][2][3]
			+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][2][2][4]
			                                                       
			+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][2][3][0]
			+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][2][3][1]
			+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][2][3][2]
			+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][2][3][3]
			+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][2][3][4]
			                                                       
			+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][2][4][0]
			+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][2][4][1]
			+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][2][4][2]
			+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][2][4][3]
			+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][2][4][4];
			
			//temp = relu(temp + net->bias0_1[2]);
			temp += net->bias0_1[2];
			if (temp < 0)
				temp = 0;
			
			features->layer1[2][o0][o1] = temp;
			
			
			
			temp = features->layer1[3][o0][o1];

			temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][3][0][0]
			+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][3][0][1]
			+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][3][0][2]
			+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][3][0][3]
			+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][3][0][4]
			                                                       
			+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][3][1][0]
			+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][3][1][1]
			+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][3][1][2]
			+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][3][1][3]
			+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][3][1][4]
			                                                        
			+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][3][2][0]
			+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][3][2][1]
			+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][3][2][2]
			+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][3][2][3]
			+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][3][2][4]
			                                                        
			+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][3][3][0]
			+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][3][3][1]
			+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][3][3][2]
			+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][3][3][3]
			+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][3][3][4]
			                                                      
			+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][3][4][0]
			+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][3][4][1]
			+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][3][4][2]
			+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][3][4][3]
			+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][3][4][4];
			
			//temp = relu(temp + net->bias0_1[3]);
			temp += net->bias0_1[3];
			if (temp < 0)
				temp = 0;
			
			features->layer1[3][o0][o1] = temp;
			
			
			
			temp = features->layer1[4][o0][o1];

			temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][4][0][0]
			+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][4][0][1]
			+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][4][0][2]
			+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][4][0][3]
			+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][4][0][4]
			                                                       
			+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][4][1][0]
			+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][4][1][1]
			+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][4][1][2]
			+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][4][1][3]
			+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][4][1][4]
			                                                     
			+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][4][2][0]
			+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][4][2][1]
			+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][4][2][2]
			+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][4][2][3]
			+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][4][2][4]
		                                                        
			+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][4][3][0]
			+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][4][3][1]
			+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][4][3][2]
			+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][4][3][3]
			+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][4][3][4]
			                                                       
			+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][4][4][0]
			+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][4][4][1]
			+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][4][4][2]
			+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][4][4][3]
			+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][4][4][4];
			
			//temp = relu(temp + net->bias0_1[4]);
			temp += net->bias0_1[4];
			if (temp < 0)
				temp = 0;
			
			features->layer1[4][o0][o1] = temp;
			
			
			
			temp = features->layer1[5][o0][o1];

			temp += features->input[0][o0 + 0][o1 + 0] * net->weight0_1[0][5][0][0]
			+ features->input[0][o0 + 0][o1 + 1] * net->weight0_1[0][5][0][1]
			+ features->input[0][o0 + 0][o1 + 2] * net->weight0_1[0][5][0][2]
			+ features->input[0][o0 + 0][o1 + 3] * net->weight0_1[0][5][0][3]
			+ features->input[0][o0 + 0][o1 + 4] * net->weight0_1[0][5][0][4]
		                                                        
			+ features->input[0][o0 + 1][o1 + 0] * net->weight0_1[0][5][1][0]
			+ features->input[0][o0 + 1][o1 + 1] * net->weight0_1[0][5][1][1]
			+ features->input[0][o0 + 1][o1 + 2] * net->weight0_1[0][5][1][2]
			+ features->input[0][o0 + 1][o1 + 3] * net->weight0_1[0][5][1][3]
			+ features->input[0][o0 + 1][o1 + 4] * net->weight0_1[0][5][1][4]
			                                                      
			+ features->input[0][o0 + 2][o1 + 0] * net->weight0_1[0][5][2][0]
			+ features->input[0][o0 + 2][o1 + 1] * net->weight0_1[0][5][2][1]
			+ features->input[0][o0 + 2][o1 + 2] * net->weight0_1[0][5][2][2]
			+ features->input[0][o0 + 2][o1 + 3] * net->weight0_1[0][5][2][3]
			+ features->input[0][o0 + 2][o1 + 4] * net->weight0_1[0][5][2][4]
			                                                       
			+ features->input[0][o0 + 3][o1 + 0] * net->weight0_1[0][5][3][0]
			+ features->input[0][o0 + 3][o1 + 1] * net->weight0_1[0][5][3][1]
			+ features->input[0][o0 + 3][o1 + 2] * net->weight0_1[0][5][3][2]
			+ features->input[0][o0 + 3][o1 + 3] * net->weight0_1[0][5][3][3]
			+ features->input[0][o0 + 3][o1 + 4] * net->weight0_1[0][5][3][4]
		                                                       
			+ features->input[0][o0 + 4][o1 + 0] * net->weight0_1[0][5][4][0]
			+ features->input[0][o0 + 4][o1 + 1] * net->weight0_1[0][5][4][1]
			+ features->input[0][o0 + 4][o1 + 2] * net->weight0_1[0][5][4][2]
			+ features->input[0][o0 + 4][o1 + 3] * net->weight0_1[0][5][4][3]
			+ features->input[0][o0 + 4][o1 + 4] * net->weight0_1[0][5][4][4];
			
			//temp = relu(temp + net->bias0_1[5]);
			temp += net->bias0_1[5];
			if (temp < 0)
				temp = 0;
			
			features->layer1[5][o0][o1] = temp;
		}
				

	// max pooling 

	for (uint8 o0 = 0; o0 < 14; o0++)
		for (uint8 o1 = 0; o1 < 14; o1++)
		{
			int x0 = 0, x1 = 0, ismax = 0, tempint = 0, o1Times2 = o1 * 2, o0Times2 = o0 * 2;
			ismax = features->layer1[0][o0 * 2][o1 * 2] < features->layer1[0][o0Times2][o1Times2 + 1];
			x1 = ismax;
			ismax = features->layer1[0][o0Times2 + 1][o1Times2] > features->layer1[0][o0Times2 ][o1Times2 + ismax];
			//tempint = features->layer1[0][o0Times2 + 1][o1Times2 + 1];
			x0 = ismax;
			x1 += ismax * (0 - x1);
			ismax = features->layer1[0][o0Times2 + 1][o1Times2 + 1] > features->layer1[0][o0Times2 + ismax][o1Times2 + x1];
			x0 += ismax * (1 - x0);
			x1 += ismax * (1 - x1);
			features->layer2[0][o0][o1] = features->layer1[0][o0Times2 + x0][o1Times2 + x1];
			
			ismax = features->layer1[1][o0 * 2][o1 * 2] < features->layer1[1][o0Times2][o1Times2 + 1];
			x1 = ismax;
			ismax = features->layer1[1][o0Times2 + 1][o1Times2] > features->layer1[1][o0Times2 ][o1Times2 + ismax];
			//tempint = features->layer1[1][o0Times2 + 1][o1Times2 + 1];
			x0 = ismax;
			x1 += ismax * (0 - x1);
			ismax = features->layer1[1][o0Times2 + 1][o1Times2 + 1] > features->layer1[1][o0Times2 + ismax][o1Times2 + x1];
			x0 += ismax * (1 - x0);
			x1 += ismax * (1 - x1);
			features->layer2[1][o0][o1] = features->layer1[1][o0Times2 + x0][o1Times2 + x1];
			
			ismax = features->layer1[2][o0 * 2][o1 * 2] < features->layer1[2][o0Times2][o1Times2 + 1];
			x1 = ismax;
			ismax = features->layer1[2][o0Times2 + 1][o1Times2] > features->layer1[2][o0Times2 ][o1Times2 + ismax];
			//tempint = features->layer1[2][o0Times2 + 1][o1Times2 + 1];
			x0 = ismax;
			x1 += ismax * (0 - x1);
			ismax = features->layer1[2][o0Times2 + 1][o1Times2 + 1] > features->layer1[2][o0Times2 + ismax][o1Times2 + x1];
			x0 += ismax * (1 - x0);
			x1 += ismax * (1 - x1);
			features->layer2[2][o0][o1] = features->layer1[2][o0Times2 + x0][o1Times2 + x1];
			
			ismax = features->layer1[3][o0 * 2][o1 * 2] < features->layer1[3][o0Times2][o1Times2 + 1];
			x1 = ismax;
			ismax = features->layer1[3][o0Times2 + 1][o1Times2] > features->layer1[3][o0Times2 ][o1Times2 + ismax];
			//tempint = features->layer1[3][o0Times2 + 1][o1Times2 + 1];
			x0 = ismax;
			x1 += ismax * (0 - x1);
			ismax = features->layer1[3][o0Times2 + 1][o1Times2 + 1] > features->layer1[3][o0Times2 + ismax][o1Times2 + x1];
			x0 += ismax * (1 - x0);
			x1 += ismax * (1 - x1);
			features->layer2[3][o0][o1] = features->layer1[3][o0Times2 + x0][o1Times2 + x1];
			
			ismax = features->layer1[4][o0 * 2][o1 * 2] < features->layer1[4][o0Times2][o1Times2 + 1];
			x1 = ismax;
			ismax = features->layer1[4][o0Times2 + 1][o1Times2] > features->layer1[4][o0Times2 ][o1Times2 + ismax];
			//tempint = features->layer1[4][o0Times2 + 1][o1Times2 + 1];
			x0 = ismax;
			x1 += ismax * (0 - x1);
			ismax = features->layer1[4][o0Times2 + 1][o1Times2 + 1] > features->layer1[4][o0Times2 + ismax][o1Times2 + x1];
			x0 += ismax * (1 - x0);
			x1 += ismax * (1 - x1);
			features->layer2[4][o0][o1] = features->layer1[4][o0Times2 + x0][o1Times2 + x1];
			
			ismax = features->layer1[5][o0 * 2][o1 * 2] < features->layer1[5][o0Times2][o1Times2 + 1];
			x1 = ismax;
			ismax = features->layer1[5][o0Times2 + 1][o1Times2] > features->layer1[5][o0Times2 ][o1Times2 + ismax];
			//tempint = features->layer1[5][o0Times2 + 1][o1Times2 + 1];
			x0 = ismax;
			x1 += ismax * (0 - x1);
			ismax = features->layer1[5][o0Times2 + 1][o1Times2 + 1] > features->layer1[5][o0Times2 + ismax][o1Times2 + x1];
			x0 += ismax * (1 - x0);
			x1 += ismax * (1 - x1);
			features->layer2[5][o0][o1] = features->layer1[5][o0Times2 + x0][o1Times2 + x1];	
		}
		

	// convolution
	double tempBias = 0.0;
	for (uint8 y = 0; y < 16; y++)
	{
		tempBias = net->bias2_3[y];
		for (uint8 o0 = 0; o0 < 10; o0++)
			for (uint8 o1 = 0; o1 < 10; o1++)
			{
				temp = features->layer3[y][o0][o1];
				
				temp += features->layer2[0][o0 + 0][o1 + 0] * net->weight2_3[0][y][0][0]
				+ features->layer2[0][o0 + 0][o1 + 1] * net->weight2_3[0][y][0][1]
				+ features->layer2[0][o0 + 0][o1 + 2] * net->weight2_3[0][y][0][2]
				+ features->layer2[0][o0 + 0][o1 + 3] * net->weight2_3[0][y][0][3]
				+ features->layer2[0][o0 + 0][o1 + 4] * net->weight2_3[0][y][0][4]
				+ features->layer2[0][o0 + 1][o1 + 0] * net->weight2_3[0][y][1][0]
				+ features->layer2[0][o0 + 1][o1 + 1] * net->weight2_3[0][y][1][1]
				+ features->layer2[0][o0 + 1][o1 + 2] * net->weight2_3[0][y][1][2]
				+ features->layer2[0][o0 + 1][o1 + 3] * net->weight2_3[0][y][1][3]
				+ features->layer2[0][o0 + 1][o1 + 4] * net->weight2_3[0][y][1][4]
				+ features->layer2[0][o0 + 2][o1 + 0] * net->weight2_3[0][y][2][0]
				+ features->layer2[0][o0 + 2][o1 + 1] * net->weight2_3[0][y][2][1]
				+ features->layer2[0][o0 + 2][o1 + 2] * net->weight2_3[0][y][2][2]
				+ features->layer2[0][o0 + 2][o1 + 3] * net->weight2_3[0][y][2][3]
				+ features->layer2[0][o0 + 2][o1 + 4] * net->weight2_3[0][y][2][4]
				+ features->layer2[0][o0 + 3][o1 + 0] * net->weight2_3[0][y][3][0]
				+ features->layer2[0][o0 + 3][o1 + 1] * net->weight2_3[0][y][3][1]
				+ features->layer2[0][o0 + 3][o1 + 2] * net->weight2_3[0][y][3][2]
				+ features->layer2[0][o0 + 3][o1 + 3] * net->weight2_3[0][y][3][3]
				+ features->layer2[0][o0 + 3][o1 + 4] * net->weight2_3[0][y][3][4]
				+ features->layer2[0][o0 + 4][o1 + 0] * net->weight2_3[0][y][4][0]
				+ features->layer2[0][o0 + 4][o1 + 1] * net->weight2_3[0][y][4][1]
				+ features->layer2[0][o0 + 4][o1 + 2] * net->weight2_3[0][y][4][2]
				+ features->layer2[0][o0 + 4][o1 + 3] * net->weight2_3[0][y][4][3]
				+ features->layer2[0][o0 + 4][o1 + 4] * net->weight2_3[0][y][4][4];
				
				temp += features->layer2[1][o0 + 0][o1 + 0] * net->weight2_3[1][y][0][0]
				+ features->layer2[1][o0 + 0][o1 + 1] * net->weight2_3[1][y][0][1]
				+ features->layer2[1][o0 + 0][o1 + 2] * net->weight2_3[1][y][0][2]
				+ features->layer2[1][o0 + 0][o1 + 3] * net->weight2_3[1][y][0][3]
				+ features->layer2[1][o0 + 0][o1 + 4] * net->weight2_3[1][y][0][4]
				+ features->layer2[1][o0 + 1][o1 + 0] * net->weight2_3[1][y][1][0]
				+ features->layer2[1][o0 + 1][o1 + 1] * net->weight2_3[1][y][1][1]
				+ features->layer2[1][o0 + 1][o1 + 2] * net->weight2_3[1][y][1][2]
				+ features->layer2[1][o0 + 1][o1 + 3] * net->weight2_3[1][y][1][3]
				+ features->layer2[1][o0 + 1][o1 + 4] * net->weight2_3[1][y][1][4]
				+ features->layer2[1][o0 + 2][o1 + 0] * net->weight2_3[1][y][2][0]
				+ features->layer2[1][o0 + 2][o1 + 1] * net->weight2_3[1][y][2][1]
				+ features->layer2[1][o0 + 2][o1 + 2] * net->weight2_3[1][y][2][2]
				+ features->layer2[1][o0 + 2][o1 + 3] * net->weight2_3[1][y][2][3]
				+ features->layer2[1][o0 + 2][o1 + 4] * net->weight2_3[1][y][2][4]
				+ features->layer2[1][o0 + 3][o1 + 0] * net->weight2_3[1][y][3][0]
				+ features->layer2[1][o0 + 3][o1 + 1] * net->weight2_3[1][y][3][1]
				+ features->layer2[1][o0 + 3][o1 + 2] * net->weight2_3[1][y][3][2]
				+ features->layer2[1][o0 + 3][o1 + 3] * net->weight2_3[1][y][3][3]
				+ features->layer2[1][o0 + 3][o1 + 4] * net->weight2_3[1][y][3][4]
				+ features->layer2[1][o0 + 4][o1 + 0] * net->weight2_3[1][y][4][0]
				+ features->layer2[1][o0 + 4][o1 + 1] * net->weight2_3[1][y][4][1]
				+ features->layer2[1][o0 + 4][o1 + 2] * net->weight2_3[1][y][4][2]
				+ features->layer2[1][o0 + 4][o1 + 3] * net->weight2_3[1][y][4][3]
				+ features->layer2[1][o0 + 4][o1 + 4] * net->weight2_3[1][y][4][4];
				
				temp += features->layer2[2][o0 + 0][o1 + 0] * net->weight2_3[2][y][0][0]
				+ features->layer2[2][o0 + 0][o1 + 1] * net->weight2_3[2][y][0][1]
				+ features->layer2[2][o0 + 0][o1 + 2] * net->weight2_3[2][y][0][2]
				+ features->layer2[2][o0 + 0][o1 + 3] * net->weight2_3[2][y][0][3]
				+ features->layer2[2][o0 + 0][o1 + 4] * net->weight2_3[2][y][0][4]
				+ features->layer2[2][o0 + 1][o1 + 0] * net->weight2_3[2][y][1][0]
				+ features->layer2[2][o0 + 1][o1 + 1] * net->weight2_3[2][y][1][1]
				+ features->layer2[2][o0 + 1][o1 + 2] * net->weight2_3[2][y][1][2]
				+ features->layer2[2][o0 + 1][o1 + 3] * net->weight2_3[2][y][1][3]
				+ features->layer2[2][o0 + 1][o1 + 4] * net->weight2_3[2][y][1][4]
				+ features->layer2[2][o0 + 2][o1 + 0] * net->weight2_3[2][y][2][0]
				+ features->layer2[2][o0 + 2][o1 + 1] * net->weight2_3[2][y][2][1]
				+ features->layer2[2][o0 + 2][o1 + 2] * net->weight2_3[2][y][2][2]
				+ features->layer2[2][o0 + 2][o1 + 3] * net->weight2_3[2][y][2][3]
				+ features->layer2[2][o0 + 2][o1 + 4] * net->weight2_3[2][y][2][4]
				+ features->layer2[2][o0 + 3][o1 + 0] * net->weight2_3[2][y][3][0]
				+ features->layer2[2][o0 + 3][o1 + 1] * net->weight2_3[2][y][3][1]
				+ features->layer2[2][o0 + 3][o1 + 2] * net->weight2_3[2][y][3][2]
				+ features->layer2[2][o0 + 3][o1 + 3] * net->weight2_3[2][y][3][3]
				+ features->layer2[2][o0 + 3][o1 + 4] * net->weight2_3[2][y][3][4]
				+ features->layer2[2][o0 + 4][o1 + 0] * net->weight2_3[2][y][4][0]
				+ features->layer2[2][o0 + 4][o1 + 1] * net->weight2_3[2][y][4][1]
				+ features->layer2[2][o0 + 4][o1 + 2] * net->weight2_3[2][y][4][2]
				+ features->layer2[2][o0 + 4][o1 + 3] * net->weight2_3[2][y][4][3]
				+ features->layer2[2][o0 + 4][o1 + 4] * net->weight2_3[2][y][4][4];
				
				temp += features->layer2[3][o0 + 0][o1 + 0] * net->weight2_3[3][y][0][0]
				+ features->layer2[3][o0 + 0][o1 + 1] * net->weight2_3[3][y][0][1]
				+ features->layer2[3][o0 + 0][o1 + 2] * net->weight2_3[3][y][0][2]
				+ features->layer2[3][o0 + 0][o1 + 3] * net->weight2_3[3][y][0][3]
				+ features->layer2[3][o0 + 0][o1 + 4] * net->weight2_3[3][y][0][4]
				+ features->layer2[3][o0 + 1][o1 + 0] * net->weight2_3[3][y][1][0]
				+ features->layer2[3][o0 + 1][o1 + 1] * net->weight2_3[3][y][1][1]
				+ features->layer2[3][o0 + 1][o1 + 2] * net->weight2_3[3][y][1][2]
				+ features->layer2[3][o0 + 1][o1 + 3] * net->weight2_3[3][y][1][3]
				+ features->layer2[3][o0 + 1][o1 + 4] * net->weight2_3[3][y][1][4]
				+ features->layer2[3][o0 + 2][o1 + 0] * net->weight2_3[3][y][2][0]
				+ features->layer2[3][o0 + 2][o1 + 1] * net->weight2_3[3][y][2][1]
				+ features->layer2[3][o0 + 2][o1 + 2] * net->weight2_3[3][y][2][2]
				+ features->layer2[3][o0 + 2][o1 + 3] * net->weight2_3[3][y][2][3]
				+ features->layer2[3][o0 + 2][o1 + 4] * net->weight2_3[3][y][2][4]
				+ features->layer2[3][o0 + 3][o1 + 0] * net->weight2_3[3][y][3][0]
				+ features->layer2[3][o0 + 3][o1 + 1] * net->weight2_3[3][y][3][1]
				+ features->layer2[3][o0 + 3][o1 + 2] * net->weight2_3[3][y][3][2]
				+ features->layer2[3][o0 + 3][o1 + 3] * net->weight2_3[3][y][3][3]
				+ features->layer2[3][o0 + 3][o1 + 4] * net->weight2_3[3][y][3][4]
				+ features->layer2[3][o0 + 4][o1 + 0] * net->weight2_3[3][y][4][0]
				+ features->layer2[3][o0 + 4][o1 + 1] * net->weight2_3[3][y][4][1]
				+ features->layer2[3][o0 + 4][o1 + 2] * net->weight2_3[3][y][4][2]
				+ features->layer2[3][o0 + 4][o1 + 3] * net->weight2_3[3][y][4][3]
				+ features->layer2[3][o0 + 4][o1 + 4] * net->weight2_3[3][y][4][4];
				
				temp += features->layer2[4][o0 + 0][o1 + 0] * net->weight2_3[4][y][0][0]
				+ features->layer2[4][o0 + 0][o1 + 1] * net->weight2_3[4][y][0][1]
				+ features->layer2[4][o0 + 0][o1 + 2] * net->weight2_3[4][y][0][2]
				+ features->layer2[4][o0 + 0][o1 + 3] * net->weight2_3[4][y][0][3]
				+ features->layer2[4][o0 + 0][o1 + 4] * net->weight2_3[4][y][0][4]
				+ features->layer2[4][o0 + 1][o1 + 0] * net->weight2_3[4][y][1][0]
				+ features->layer2[4][o0 + 1][o1 + 1] * net->weight2_3[4][y][1][1]
				+ features->layer2[4][o0 + 1][o1 + 2] * net->weight2_3[4][y][1][2]
				+ features->layer2[4][o0 + 1][o1 + 3] * net->weight2_3[4][y][1][3]
				+ features->layer2[4][o0 + 1][o1 + 4] * net->weight2_3[4][y][1][4]
				+ features->layer2[4][o0 + 2][o1 + 0] * net->weight2_3[4][y][2][0]
				+ features->layer2[4][o0 + 2][o1 + 1] * net->weight2_3[4][y][2][1]
				+ features->layer2[4][o0 + 2][o1 + 2] * net->weight2_3[4][y][2][2]
				+ features->layer2[4][o0 + 2][o1 + 3] * net->weight2_3[4][y][2][3]
				+ features->layer2[4][o0 + 2][o1 + 4] * net->weight2_3[4][y][2][4]
				+ features->layer2[4][o0 + 3][o1 + 0] * net->weight2_3[4][y][3][0]
				+ features->layer2[4][o0 + 3][o1 + 1] * net->weight2_3[4][y][3][1]
				+ features->layer2[4][o0 + 3][o1 + 2] * net->weight2_3[4][y][3][2]
				+ features->layer2[4][o0 + 3][o1 + 3] * net->weight2_3[4][y][3][3]
				+ features->layer2[4][o0 + 3][o1 + 4] * net->weight2_3[4][y][3][4]
				+ features->layer2[4][o0 + 4][o1 + 0] * net->weight2_3[4][y][4][0]
				+ features->layer2[4][o0 + 4][o1 + 1] * net->weight2_3[4][y][4][1]
				+ features->layer2[4][o0 + 4][o1 + 2] * net->weight2_3[4][y][4][2]
				+ features->layer2[4][o0 + 4][o1 + 3] * net->weight2_3[4][y][4][3]
				+ features->layer2[4][o0 + 4][o1 + 4] * net->weight2_3[4][y][4][4];
				
				
				temp += features->layer2[5][o0 + 0][o1 + 0] * net->weight2_3[5][y][0][0]
					+ features->layer2[5][o0 + 0][o1 + 1] * net->weight2_3[5][y][0][1]
					+ features->layer2[5][o0 + 0][o1 + 2] * net->weight2_3[5][y][0][2]
					+ features->layer2[5][o0 + 0][o1 + 3] * net->weight2_3[5][y][0][3]
					+ features->layer2[5][o0 + 0][o1 + 4] * net->weight2_3[5][y][0][4]
					+ features->layer2[5][o0 + 1][o1 + 0] * net->weight2_3[5][y][1][0]
					+ features->layer2[5][o0 + 1][o1 + 1] * net->weight2_3[5][y][1][1]
					+ features->layer2[5][o0 + 1][o1 + 2] * net->weight2_3[5][y][1][2]
					+ features->layer2[5][o0 + 1][o1 + 3] * net->weight2_3[5][y][1][3]
					+ features->layer2[5][o0 + 1][o1 + 4] * net->weight2_3[5][y][1][4]
					+ features->layer2[5][o0 + 2][o1 + 0] * net->weight2_3[5][y][2][0]
					+ features->layer2[5][o0 + 2][o1 + 1] * net->weight2_3[5][y][2][1]
					+ features->layer2[5][o0 + 2][o1 + 2] * net->weight2_3[5][y][2][2]
					+ features->layer2[5][o0 + 2][o1 + 3] * net->weight2_3[5][y][2][3]
					+ features->layer2[5][o0 + 2][o1 + 4] * net->weight2_3[5][y][2][4]
					+ features->layer2[5][o0 + 3][o1 + 0] * net->weight2_3[5][y][3][0]
					+ features->layer2[5][o0 + 3][o1 + 1] * net->weight2_3[5][y][3][1]
					+ features->layer2[5][o0 + 3][o1 + 2] * net->weight2_3[5][y][3][2]
					+ features->layer2[5][o0 + 3][o1 + 3] * net->weight2_3[5][y][3][3]
					+ features->layer2[5][o0 + 3][o1 + 4] * net->weight2_3[5][y][3][4]	
					+ features->layer2[5][o0 + 4][o1 + 0] * net->weight2_3[5][y][4][0]
					+ features->layer2[5][o0 + 4][o1 + 1] * net->weight2_3[5][y][4][1]
					+ features->layer2[5][o0 + 4][o1 + 2] * net->weight2_3[5][y][4][2]
					+ features->layer2[5][o0 + 4][o1 + 3] * net->weight2_3[5][y][4][3]
					+ features->layer2[5][o0 + 4][o1 + 4] * net->weight2_3[5][y][4][4];

				features->layer3[y][o0][o1] = temp;
				
				temp += tempBias;
					if (temp < 0)
						temp = 0;
					
				features->layer3[y][o0][o1] = temp;
			}
	}
		


	// max pooling
	for (uint8 i = 0; i < 16; i++)
	{	
		int x0 = 0, x1 = 0, ismax; double tempD;
		x1 = features->layer3[i][0][1] > features->layer3[i][0][0];
		x0 = features->layer3[i][1][0] > features->layer3[i][0][x1];
		tempD = features->layer3[i][1][1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][x0][x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][0][0] = features->layer3[i][x0][x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][0][3] > features->layer3[i][0][2];
		x0 = features->layer3[i][1][2] > features->layer3[i][0][2 + x1];
		tempD = features->layer3[i][1][3];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][x0][2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][0][1] = features->layer3[i][x0][2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][0][5] > features->layer3[i][0][4];
		x0 = features->layer3[i][1][4] > features->layer3[i][0][4 + x1];
		tempD = features->layer3[i][0 * 2 + 1][5];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][0 * 2 + x0][2 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][0][2] = features->layer3[i][0 * 2 + x0][2 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][0 * 2][3 * 2 + 1] > features->layer3[i][0 * 2][3 * 2];
		x0 = features->layer3[i][0 * 2 + 1][3 * 2] > features->layer3[i][0 * 2][3 * 2 + x1];
		tempD = features->layer3[i][0 * 2 + 1][3 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][0 * 2 + x0][3 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][0][3] = features->layer3[i][0 * 2 + x0][3 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][0 * 2][4 * 2 + 1] > features->layer3[i][0 * 2][4 * 2];
		x0 = features->layer3[i][0 * 2 + 1][4 * 2] > features->layer3[i][0 * 2][4 * 2 + x1];
		tempD = features->layer3[i][0 * 2 + 1][4 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][0 * 2 + x0][4 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][0][4] = features->layer3[i][0 * 2 + x0][4 * 2 + x1];
		
		
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][1 * 2][0 * 2 + 1] > features->layer3[i][1 * 2][0 * 2];
		x0 = features->layer3[i][1 * 2 + 1][0 * 2] > features->layer3[i][1 * 2][0 * 2 + x1];
		tempD = features->layer3[i][1 * 2 + 1][0 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][1 * 2 + x0][0 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][1][0] = features->layer3[i][1 * 2 + x0][0 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][1 * 2][1 * 2 + 1] > features->layer3[i][1 * 2][1 * 2];
		x0 = features->layer3[i][1 * 2 + 1][1 * 2] > features->layer3[i][1 * 2][1 * 2 + x1];
		tempD = features->layer3[i][1 * 2 + 1][1 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][1 * 2 + x0][1 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][1][1] = features->layer3[i][1 * 2 + x0][1 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][1 * 2][2 * 2 + 1] > features->layer3[i][1 * 2][2 * 2];
		x0 = features->layer3[i][1 * 2 + 1][2 * 2] > features->layer3[i][1 * 2][2 * 2 + x1];
		tempD = features->layer3[i][1 * 2 + 1][2 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][1 * 2 + x0][2 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][1][2] = features->layer3[i][1 * 2 + x0][2 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][1 * 2][3 * 2 + 1] > features->layer3[i][1 * 2][3 * 2];
		x0 = features->layer3[i][1 * 2 + 1][3 * 2] > features->layer3[i][1 * 2][3 * 2 + x1];
		tempD = features->layer3[i][1 * 2 + 1][3 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][1 * 2 + x0][3 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][1][3] = features->layer3[i][1 * 2 + x0][3 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][1 * 2][4 * 2 + 1] > features->layer3[i][1 * 2][4 * 2];
		x0 = features->layer3[i][1 * 2 + 1][4 * 2] > features->layer3[i][1 * 2][4 * 2 + x1];
		tempD = features->layer3[i][1 * 2 + 1][4 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][1 * 2 + x0][4 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][1][4] = features->layer3[i][1 * 2 + x0][4 * 2 + x1];
		
		
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][2 * 2][0 * 2 + 1] > features->layer3[i][2 * 2][0 * 2];
		x0 = features->layer3[i][2 * 2 + 1][0 * 2] > features->layer3[i][2 * 2][0 * 2 + x1];
		tempD = features->layer3[i][2 * 2 + 1][0 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][2 * 2 + x0][0 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][2][0] = features->layer3[i][2 * 2 + x0][0 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][2 * 2][1 * 2 + 1] > features->layer3[i][2 * 2][1 * 2];
		x0 = features->layer3[i][2 * 2 + 1][1 * 2] > features->layer3[i][2 * 2][1 * 2 + x1];
		tempD = features->layer3[i][2 * 2 + 1][1 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][2 * 2 + x0][1 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][2][1] = features->layer3[i][2 * 2 + x0][1 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][2 * 2][2 * 2 + 1] > features->layer3[i][2 * 2][2 * 2];
		x0 = features->layer3[i][2 * 2 + 1][2 * 2] > features->layer3[i][2 * 2][2 * 2 + x1];
		tempD = features->layer3[i][2 * 2 + 1][2 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][2 * 2 + x0][2 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][2][2] = features->layer3[i][2 * 2 + x0][2 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][2 * 2][3 * 2 + 1] > features->layer3[i][2 * 2][3 * 2];
		x0 = features->layer3[i][2 * 2 + 1][3 * 2] > features->layer3[i][2 * 2][3 * 2 + x1];
		tempD = features->layer3[i][2 * 2 + 1][3 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][2 * 2 + x0][3 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][2][3] = features->layer3[i][2 * 2 + x0][3 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][2 * 2][4 * 2 + 1] > features->layer3[i][2 * 2][4 * 2];
		x0 = features->layer3[i][2 * 2 + 1][4 * 2] > features->layer3[i][2 * 2][4 * 2 + x1];
		tempD = features->layer3[i][2 * 2 + 1][4 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][2 * 2 + x0][4 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][2][4] = features->layer3[i][2 * 2 + x0][4 * 2 + x1];
		
		
		
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][3 * 2][0 * 2 + 1] > features->layer3[i][3 * 2][0 * 2];
		x0 = features->layer3[i][3 * 2 + 1][0 * 2] > features->layer3[i][3 * 2][0 * 2 + x1];
		tempD = features->layer3[i][3 * 2 + 1][0 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][3 * 2 + x0][0 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][3][0] = features->layer3[i][3 * 2 + x0][0 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][3 * 2][1 * 2 + 1] > features->layer3[i][3 * 2][1 * 2];
		x0 = features->layer3[i][3 * 2 + 1][1 * 2] > features->layer3[i][3 * 2][1 * 2 + x1];
		tempD = features->layer3[i][3 * 2 + 1][1 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][3 * 2 + x0][1 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][3][1] = features->layer3[i][3 * 2 + x0][1 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][3 * 2][2 * 2 + 1] > features->layer3[i][3 * 2][2 * 2];
		x0 = features->layer3[i][3 * 2 + 1][2 * 2] > features->layer3[i][3 * 2][2 * 2 + x1];
		tempD = features->layer3[i][3 * 2 + 1][2 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][3 * 2 + x0][2 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][3][2] = features->layer3[i][3 * 2 + x0][2 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][3 * 2][3 * 2 + 1] > features->layer3[i][3 * 2][3 * 2];
		x0 = features->layer3[i][3 * 2 + 1][3 * 2] > features->layer3[i][3 * 2][3 * 2 + x1];
		tempD = features->layer3[i][3 * 2 + 1][3 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][3 * 2 + x0][3 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][3][3] = features->layer3[i][3 * 2 + x0][3 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][3 * 2][4 * 2 + 1] > features->layer3[i][3 * 2][4 * 2];
		x0 = features->layer3[i][3 * 2 + 1][4 * 2] > features->layer3[i][3 * 2][4 * 2 + x1];
		tempD = features->layer3[i][3 * 2 + 1][4 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][3 * 2 + x0][4 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][3][4] = features->layer3[i][3 * 2 + x0][4 * 2 + x1];
		
		
		
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][4 * 2][0 * 2 + 1] > features->layer3[i][4 * 2][0 * 2];
		x0 = features->layer3[i][4 * 2 + 1][0 * 2] > features->layer3[i][4 * 2][0 * 2 + x1];
		tempD = features->layer3[i][4 * 2 + 1][0 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][4 * 2 + x0][0 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][4][0] = features->layer3[i][4 * 2 + x0][0 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][4 * 2][1 * 2 + 1] > features->layer3[i][4 * 2][1 * 2];
		x0 = features->layer3[i][4 * 2 + 1][1 * 2] > features->layer3[i][4 * 2][1 * 2 + x1];
		tempD = features->layer3[i][4 * 2 + 1][1 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][4 * 2 + x0][1 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][4][1] = features->layer3[i][4 * 2 + x0][1 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][4 * 2][2 * 2 + 1] > features->layer3[i][4 * 2][2 * 2];
		x0 = features->layer3[i][4 * 2 + 1][2 * 2] > features->layer3[i][4 * 2][2 * 2 + x1];
		tempD = features->layer3[i][4 * 2 + 1][2 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][4 * 2 + x0][2 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][4][2] = features->layer3[i][4 * 2 + x0][2 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][4 * 2][3 * 2 + 1] > features->layer3[i][4 * 2][3 * 2];
		x0 = features->layer3[i][4 * 2 + 1][3 * 2] > features->layer3[i][4 * 2][3 * 2 + x1];
		tempD = features->layer3[i][4 * 2 + 1][3 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][4 * 2 + x0][3 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][4][3] = features->layer3[i][4 * 2 + x0][3 * 2 + x1];
		x0 = 0; x1 = 0;
		x1 = features->layer3[i][4 * 2][4 * 2 + 1] > features->layer3[i][4 * 2][4 * 2];
		x0 = features->layer3[i][4 * 2 + 1][4 * 2] > features->layer3[i][4 * 2][4 * 2 + x1];
		tempD = features->layer3[i][4 * 2 + 1][4 * 2 + 1];
		x1 += x0 * (- x1);	
		ismax = tempD > features->layer3[i][4 * 2 + x0][4 * 2 + x1];
		x0 += ismax * (1 - x0);
		x1 += ismax * (1 - x1);
		features->layer4[i][4][4] = features->layer3[i][4 * 2 + x0][4 * 2 + x1];
	}
	
	
	//convolution
	for (uint8 x = 0; x < 15; x++)
		for (uint8 y = 0; y < 120; y++)
		{
			temp = features->layer5[y][0][0];
			
			temp += features->layer4[x][0][0] * net->weight4_5[x][y][0][0]
			+ features->layer4[x][0][1] * net->weight4_5[x][y][0][1]
			+ features->layer4[x][0][2] * net->weight4_5[x][y][0][2]
			+ features->layer4[x][0][3] * net->weight4_5[x][y][0][3]
			+ features->layer4[x][0][4] * net->weight4_5[x][y][0][4]
			                                                                   
			+ features->layer4[x][1][0] * net->weight4_5[x][y][1][0]
			+ features->layer4[x][1][1] * net->weight4_5[x][y][1][1]
			+ features->layer4[x][1][2] * net->weight4_5[x][y][1][2]
			+ features->layer4[x][1][3] * net->weight4_5[x][y][1][3]
			+ features->layer4[x][1][4] * net->weight4_5[x][y][1][4]
			+                                              
			+ features->layer4[x][2][0] * net->weight4_5[x][y][2][0]
			+ features->layer4[x][2][1] * net->weight4_5[x][y][2][1]
			+ features->layer4[x][2][2] * net->weight4_5[x][y][2][2]
			+ features->layer4[x][2][3] * net->weight4_5[x][y][2][3]
			+ features->layer4[x][2][4] * net->weight4_5[x][y][2][4]
			                                           
			+ features->layer4[x][3][0] * net->weight4_5[x][y][3][0]
			+ features->layer4[x][3][1] * net->weight4_5[x][y][3][1]
			+ features->layer4[x][3][2] * net->weight4_5[x][y][3][2]
			+ features->layer4[x][3][3] * net->weight4_5[x][y][3][3]
			+ features->layer4[x][3][4] * net->weight4_5[x][y][3][4]
		                                             
			+ features->layer4[x][4][0] * net->weight4_5[x][y][4][0]
			+ features->layer4[x][4][1] * net->weight4_5[x][y][4][1]
			+ features->layer4[x][4][2] * net->weight4_5[x][y][4][2]
			+ features->layer4[x][4][3] * net->weight4_5[x][y][4][3]
			+ features->layer4[x][4][4] * net->weight4_5[x][y][4][4];
			
			features->layer5[y][0][0] = temp;
		}
	
	//x = 15
	for (uint8 y = 0; y < 120; y++)
		{
			temp = features->layer5[y][0][0];
			
			temp += features->layer4[15][0][0] * net->weight4_5[15][y][0][0]
			+ features->layer4[15][0][1] * net->weight4_5[15][y][0][1]
			+ features->layer4[15][0][2] * net->weight4_5[15][y][0][2]
			+ features->layer4[15][0][3] * net->weight4_5[15][y][0][3]
			+ features->layer4[15][0][4] * net->weight4_5[15][y][0][4]
			                                                                  
			+ features->layer4[15][1][0] * net->weight4_5[15][y][1][0]
			+ features->layer4[15][1][1] * net->weight4_5[15][y][1][1]
			+ features->layer4[15][1][2] * net->weight4_5[15][y][1][2]
			+ features->layer4[15][1][3] * net->weight4_5[15][y][1][3]
			+ features->layer4[15][1][4] * net->weight4_5[15][y][1][4]
			                                            
			+ features->layer4[15][2][0] * net->weight4_5[15][y][2][0]
			+ features->layer4[15][2][1] * net->weight4_5[15][y][2][1]
			+ features->layer4[15][2][2] * net->weight4_5[15][y][2][2]
			+ features->layer4[15][2][3] * net->weight4_5[15][y][2][3]
			+ features->layer4[15][2][4] * net->weight4_5[15][y][2][4]
		                                             
			+ features->layer4[15][3][0] * net->weight4_5[15][y][3][0]
			+ features->layer4[15][3][1] * net->weight4_5[15][y][3][1]
			+ features->layer4[15][3][2] * net->weight4_5[15][y][3][2]
			+ features->layer4[15][3][3] * net->weight4_5[15][y][3][3]
			+ features->layer4[15][3][4] * net->weight4_5[15][y][3][4]
			                                                    
			+ features->layer4[15][4][0] * net->weight4_5[15][y][4][0]
			+ features->layer4[15][4][1] * net->weight4_5[15][y][4][1]
			+ features->layer4[15][4][2] * net->weight4_5[15][y][4][2]
			+ features->layer4[15][4][3] * net->weight4_5[15][y][4][3]
			+ features->layer4[15][4][4] * net->weight4_5[15][y][4][4];
			
			//features->layer5[y][0][0] = temp;
			
			//features->layer5[y][0][0] = relu(temp + net->bias4_5[y]);
			temp += net->bias4_5[y];
			if (temp < 0)
				temp = 0;
			features->layer5[y][0][0] = temp;
				
		}

	
	// fully connected layer
	for (int x = 118; x > -1; x--)
	{
		temp = features->layer5[x][0][0];
		for (int y = 255; y > -1; y--)
		{
			features->layer6[y] += temp * net->weight5_6[x][y];
		}
	}
	//x = 119
	temp = features->layer5[119][0][0];
	for (int y = 255; y > -1; y--)
	{
		features->layer6[y] += temp * net->weight5_6[119][y] + net->bias5_6[y];
		if (features->layer6[y] < 0)
			features->layer6[y] = 0;
		//features->layer6[y] = relu(features->layer6[y] + net->bias5_6[y]);
	}
		


	// fully connected layer
	for (int x = 255; x > -1; --x)
	{
		temp = features->layer6[x];
		features->output[0] += temp * net->weight6_7[x][0];
		features->output[1] += temp * net->weight6_7[x][1];
		features->output[2] += temp * net->weight6_7[x][2];
		features->output[3] += temp * net->weight6_7[x][3];
		features->output[4] += temp * net->weight6_7[x][4];
		features->output[5] += temp * net->weight6_7[x][5];
		features->output[6] += temp * net->weight6_7[x][6];
		features->output[7] += temp * net->weight6_7[x][7];
		features->output[8] += temp * net->weight6_7[x][8];
		features->output[9] += temp * net->weight6_7[x][9];
	}
	
	//x = 255
	features->output[0] = features->output[0] + net->bias6_7[0];
	features->output[1] = features->output[1] + net->bias6_7[1];
	features->output[2] = features->output[2] + net->bias6_7[2];
	features->output[3] = features->output[3] + net->bias6_7[3];
	features->output[4] = features->output[4] + net->bias6_7[4];
	features->output[5] = features->output[5] + net->bias6_7[5];
	features->output[6] = features->output[6] + net->bias6_7[6];
	features->output[7] = features->output[7] + net->bias6_7[7];
	features->output[8] = features->output[8] + net->bias6_7[8];
	features->output[9] = features->output[9] + net->bias6_7[9];

	
}

void *Predict_task(void *arg)
{
	input_param *incoming_param = (input_param *) arg;
	Net *net = incoming_param->netStructure;
	image *testData = incoming_param->testData;
	uint8 *testLabel = incoming_param->testLabel;
	
	int offSet = incoming_param->offSet;
	int remainder = incoming_param->remainder;
	
	int l, p = 0;
	
	int *result = (int *) calloc (1, sizeof(int));
	*result = 0;
	
	double *output;
	double maxvalue;
	
	
	for (int i = 0; i < int(COUNT_TEST/NUM_THREADS); i++)
	{
		l = testLabel[offSet + i];
		
		Feature features = { 0 };
		
		for (int j = 0; j < 28; j++)
		{
			features.input[0][j + PADDING][0 + PADDING] = ((double)(testData[offSet + i][j][0]) - 128) / 256;
			features.input[0][j + PADDING][1 + PADDING] = ((double)(testData[offSet + i][j][1]) - 128) / 256;
			features.input[0][j + PADDING][2 + PADDING] = ((double)(testData[offSet + i][j][2]) - 128) / 256;
			features.input[0][j + PADDING][3 + PADDING] = ((double)(testData[offSet + i][j][3]) - 128) / 256;
			features.input[0][j + PADDING][4 + PADDING] = ((double)(testData[offSet + i][j][4]) - 128) / 256;
			features.input[0][j + PADDING][5 + PADDING] = ((double)(testData[offSet + i][j][5]) - 128) / 256;
			features.input[0][j + PADDING][6 + PADDING] = ((double)(testData[offSet + i][j][6]) - 128) / 256;
			features.input[0][j + PADDING][7 + PADDING] = ((double)(testData[offSet + i][j][7]) - 128) / 256;
			features.input[0][j + PADDING][8 + PADDING] = ((double)(testData[offSet + i][j][8]) - 128) / 256;
			features.input[0][j + PADDING][9 + PADDING] = ((double)(testData[offSet + i][j][9]) - 128) / 256;
			features.input[0][j + PADDING][10 + PADDING] = ((double)(testData[offSet + i][j][10]) - 128) / 256;
			features.input[0][j + PADDING][11 + PADDING] = ((double)(testData[offSet + i][j][11]) - 128) / 256;
			features.input[0][j + PADDING][12 + PADDING] = ((double)(testData[offSet + i][j][12]) - 128) / 256;
			features.input[0][j + PADDING][13 + PADDING] = ((double)(testData[offSet + i][j][13]) - 128) / 256;
			features.input[0][j + PADDING][14 + PADDING] = ((double)(testData[offSet + i][j][14]) - 128) / 256;
			features.input[0][j + PADDING][15 + PADDING] = ((double)(testData[offSet + i][j][15]) - 128) / 256;
			features.input[0][j + PADDING][16 + PADDING] = ((double)(testData[offSet + i][j][16]) - 128) / 256;
			features.input[0][j + PADDING][17 + PADDING] = ((double)(testData[offSet + i][j][17]) - 128) / 256;
			features.input[0][j + PADDING][18 + PADDING] = ((double)(testData[offSet + i][j][18]) - 128) / 256;
			features.input[0][j + PADDING][19 + PADDING] = ((double)(testData[offSet + i][j][19]) - 128) / 256;
			features.input[0][j + PADDING][20 + PADDING] = ((double)(testData[offSet + i][j][20]) - 128) / 256;
			features.input[0][j + PADDING][21 + PADDING] = ((double)(testData[offSet + i][j][21]) - 128) / 256;
			features.input[0][j + PADDING][22 + PADDING] = ((double)(testData[offSet + i][j][22]) - 128) / 256;
			features.input[0][j + PADDING][23 + PADDING] = ((double)(testData[offSet + i][j][23]) - 128) / 256;
			features.input[0][j + PADDING][24 + PADDING] = ((double)(testData[offSet + i][j][24]) - 128) / 256;
			features.input[0][j + PADDING][25 + PADDING] = ((double)(testData[offSet + i][j][25]) - 128) / 256;
			features.input[0][j + PADDING][26 + PADDING] = ((double)(testData[offSet + i][j][26]) - 128) / 256;
			features.input[0][j + PADDING][27 + PADDING] = ((double)(testData[offSet + i][j][27]) - 128) / 256;
		}
			
		forward(net, &features);
		output = (double *)features.output;
		maxvalue = *output;
		p = 0;
		for (int i = 1; i < 10; i++)
			if (output[i] > maxvalue)
			{
				maxvalue = output[i];
				p = i;
			}
		if (l == p)
			*result += 1;
	}
	
	if ((NUM_THREADS - 1) * int(COUNT_TEST/NUM_THREADS) == offSet)
	{
		for (int i = 0; i < remainder; i++)
		{
			l = testLabel[COUNT_TEST - i - 1];
			
			Feature features = { 0 };
			for (int j = 0; j < 28; j++)
				for (int k = 0; k < 28; k++){
					features.input[0][j + PADDING][k + PADDING] = ((double)(testData[COUNT_TEST - i - 1][j][k]) - 128) / 256;
				}
			forward(net, &features);
			output = (double *)features.output;
			maxvalue = *output;
			p = 0;
			for (int i = 1; i < 10; i++)
				if (output[i] > maxvalue)
				{
					maxvalue = output[i];
					p = i;
				}			
			
			if (l == p)
				*result += 1;
		}
	}
	
	return result;
	
}

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}


int main()
{
	int i, l, p;
	int totalCorrect = 0;
	
    //Timer For GCC
	struct timeval t1, t2;

	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

	// reading test data (images)
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!\n");
		free(test_data);
		free(test_label);
		return 1;
	}

	// loading model
	Net *net = (Net *)malloc(sizeof(Net));
	FILE *fp = fopen("trained.model", "rb");
	if (!fp){
        printf("ERROR!!!\n");
        return 1;
	}

	fread(net, sizeof(Net), 1, fp);
	fclose(fp);
    printf("Please wait.....\n");
	//region of interest for acceleration

	//Timer For GCC
	gettimeofday(&t1, NULL);

	
	pthread_t tid[NUM_THREADS];
	void *thread_result;
	int *return_value;
	
	input_param *inputs = (input_param *)calloc(NUM_THREADS, sizeof(input_param));

	int remainder = COUNT_TEST - ((NUM_THREADS-1)*int(COUNT_TEST/NUM_THREADS) + int(COUNT_TEST/NUM_THREADS));
	
	for (int i = 0; i < NUM_THREADS; i++)
		inputs[i] = { i * int(COUNT_TEST/NUM_THREADS),  remainder, net, test_label, test_data };
	
	
	for (int i = 0; i < NUM_THREADS; i++)
		pthread_create (&tid[i], NULL, Predict_task, &inputs[i]);
	
	
	for (int i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(tid[i], &thread_result);
		return_value = (int *) thread_result;
		totalCorrect += *return_value;
	}
	
	free(return_value);
	
	// end of allowed region for modify

	//Timer For GCC
	gettimeofday(&t2, NULL);

	printf("%d / %d\n", totalCorrect, COUNT_TEST);

	//Timer For GCC
	printf("%ld seconds and %ld microseconds\n",(long)(t2.tv_sec - t1.tv_sec), (long)(t2.tv_usec - t1.tv_usec));

	free(net);
	free(test_data);
	free(test_label);
	free(inputs);

	return 0;
}

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

double relu(double x)
{
	// relu function body is allowed to modification
	if (x > 0)
		return x;
	return 0;
}

void forward(Net *net, Feature *features)
{
	// forward function body is allowed to modification
	int i, j, x, y, o0, o1, w0, w1, l0, l1;
	// convolution
	for (x = 0; x < 1; x++)
		for (y = 0; y < 6; y++)
			for (o0 = 0; o0 < 28; o0++)
				for (o1 = 0; o1 < 28; o1++)
					for (w0 = 0; w0 < 5; w0++)
						for (w1 = 0; w1 < 5; w1++)
							features->layer1[y][o0][o1] += features->input[x][o0 + w0][o1 + w1] * net->weight0_1[x][y][w0][w1];
	for (y = 0; y < 6; y++)
		for (j = 0; j < 28; j++)
			for (i = 0; i < 28; i++)
				features->layer1[y][j][i] = relu(features->layer1[y][j][i] + net->bias0_1[y]);

	// max pooling
	for(i = 0; i < 6; i++)
		for (o0 = 0; o0 < 14; o0++)
			for (o1 = 0; o1 < 14; o1++)
			{
				int x0 = 0, x1 = 0, ismax;
				for(l0 = 0; l0 < 2; l0++)
					for (l1 = 0; l1 < 2; l1++)
					{
						ismax = features->layer1[i][o0 * 2 + l0][o1 * 2 + l1] > features->layer1[i][o0 * 2 + x0][o1 * 2 + x1];
						x0 += ismax * (l0 - x0);
						x1 += ismax * (l1 - x1);
					}
				features->layer2[i][o0][o1] = features->layer1[i][o0 * 2 + x0][o1 * 2 + x1];
			}

	// convolution
	for (x = 0; x < 6; x++)
		for (y = 0; y < 16; y++)
			for (o0 = 0; o0 < 10; o0++)
				for (o1 = 0; o1 < 10; o1++)
					for (w0 = 0; w0 < 5; w0++)
						for (w1 = 0; w1 < 5; w1++)
							features->layer3[y][o0][o1] += features->layer2[x][o0 + w0][o1 + w1] * net->weight2_3[x][y][w0][w1];
	for (y = 0; y < 16; y++)
		for (j = 0; j < 10; j++)
			for (i = 0; i < 10; i++)
				features->layer3[y][j][i] = relu(features->layer3[y][j][i] + net->bias2_3[y]);


	// max pooling
	for (i = 0; i < 16; i++)
		for (o0 = 0; o0 < 5; o0++)
			for (o1 = 0; o1 < 5; o1++)
			{
				int x0 = 0, x1 = 0, ismax;
				for (l0 = 0; l0 < 2; l0++)
					for (l1 = 0; l1 < 2; l1++)
					{
						ismax = features->layer3[i][o0 * 2 + l0][o1 * 2 + l1] > features->layer3[i][o0 * 2 + x0][o1 * 2 + x1];
						x0 += ismax * (l0 - x0);
						x1 += ismax * (l1 - x1);
					}
				features->layer4[i][o0][o1] = features->layer3[i][o0 * 2 + x0][o1 * 2 + x1];
			}

	//convolution
	for (x = 0; x < 16; x++)
		for (y = 0; y < 120; y++)
			for (o0 = 0; o0 < 1; o0++)
				for (o1 = 0; o1 < 1; o1++)
					for (w0 = 0; w0 < 5; w0++)
						for (w1 = 0; w1 < 5; w1++)
							features->layer5[y][o0][o1] += features->layer4[x][o0 + w0][o1 + w1] * net->weight4_5[x][y][w0][w1];

	for (y = 0; y < 120; y++)
		for (j = 0; j < 1; j++)
			for (i = 0; i < 1; i++)
				features->layer5[y][j][i] = relu(features->layer5[y][j][i] + net->bias4_5[y]);


	// fully connected layer
	for (x = 0; x < 120; x++)
		for ( y = 0; y < 256; y++)
			features->layer6[y] += features->layer5[x][0][0] * net->weight5_6[x][y];
	for (j = 0; j < 256; j++)
		features->layer6[j] = relu(features->layer6[j] + net->bias5_6[j]);

	// fully connected layer
	for (x = 0; x < 256; x++)
		for ( y = 0; y < 10; y++)
			features->output[y] += features->layer6[x] * net->weight6_7[x][y];
	for (j = 0; j < 10; j++)
		features->output[j] = features->output[j] + net->bias6_7[j];
}

int Predict(Net *net, image input)
{
	// Predict function body is allowed to modification
	int i, j, k;
	Feature features = { 0 };

	// loading image as input
	for (j = 0; j < 28; j++)
		for (k = 0; k < 28; k++){
            features.input[0][j + PADDING][k + PADDING] = ((double)(input[j][k]) - 128) / 256;
		}


	// calculating
	forward(net, &features);

	// output decoding
	double *output = (double *)features.output;
	int result = 0;
	double maxvalue = *output;
	for (i = 1; i < 10; i++)
		if (output[i] > maxvalue)
		{
			maxvalue = output[i];
			result = i;
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
	int truePredict = 0;
    //Timer For GCC
	//struct timeval t1, t2;

    //Timer For Visual studio
	time_t *StartTime, EndTime;

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
	//gettimeofday(&t1, NULL);

	//Timer For Visual studio
	StartTime = time(NULL);

	// start of allowed region for modify
	for (i = 0; i < COUNT_TEST; i++)
	{
		l = test_label[i];
		p = Predict(net, test_data[i]);
		if (l == p)
			truePredict++;
	}
	// end of allowed region for modify

	//Timer For GCC
	//gettimeofday(&t2, NULL);

	//Timer For Visual studio
	EndTime = time(NULL);

	printf("%d / %d\n", truePredict, COUNT_TEST);

	//Timer For GCC
	//printf("%ld seconds and %ld microseconds\n",(long)(t2.tv_sec - t1.tv_sec), (long)(t2.tv_usec - t1.tv_usec));

	//Timer For Visual studio
	printf("%ld seconds and %ld microseconds\n", (long)difftime(EndTime, StartTime), 0);

	free(net);
	free(test_data);
	free(test_label);

	return 0;
}

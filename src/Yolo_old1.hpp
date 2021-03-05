#include <iostream>
#include <fstream>

//opencv libraries
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

struct yoloClass
{
	Rect boundBox;
	string classID;
	float confidence;
};

class YoloNetwork
{
	private:
		Net net;			//network
		float confThreshold;		//confidence threshold
		float nmsThreshold;		//non-maximum suppression threshold
		vector<string> classes;		//vector string classes names for coco.names
		vector<string> outputNames;	//vector string classes names for output names
		int width, height;		//width and height of input image
		vector<yoloClass> object;	//vector type yoloClass object variable
		
	public:
		YoloNetwork(const string configFile, const string weightFile, 
		const string coconames);	//constructor
		~YoloNetwork();			//destructor
		
		void CurrentFrame(Mat cap);	//Process current frame
		vector<yoloClass> outputObject();	//return current object yoloClass
};
//constructor of yolo network
YoloNetwork::YoloNetwork(const string configFile, const string weightFile, const string coconames)
{

	this->confThreshold = 0.5;	//confidence threshold for detection
	this->nmsThreshold = 0.4;	//non-maximum suppression threshold
	this->width = 416;		//width of the input image
	this->height = 416;		//height of the input image
	
	//open class file
	ifstream classes_ifs(coconames.c_str());
	string line;
	if(classes_ifs.is_open())
	{
		//get classes names
		while(getline(classes_ifs, line))
		{
			this->classes.push_back(line);
		}
	}
	//Load the network with the configuration file and weight file by YOLO
	this->net = readNetFromDarknet(configFile, weightFile);
	this->net.setPreferableBackend(DNN_BACKEND_CUDA);
	this->net.setPreferableTarget(DNN_TARGET_CUDA);
	
	//get the names of unconnected output layers from the network
	this->outputNames = this->net.getUnconnectedOutLayersNames();
}
//destructor
YoloNetwork::~YoloNetwork()
{

}
//current frame of image or webcam function to detect bounding boxes and classes
void YoloNetwork::CurrentFrame(Mat cap)
{
	//convert the image to a 4-D blob
	Mat blob;
	blobFromImage(cap, blob, 1/255.0, Size(this->width, this->height), Scalar(0,0,0), true, false);
	
	//set the network input
	net.setInput(blob);
	
	//running the network
	vector<Mat> netOut;
	net.forward(netOut, this->outputNames);
	
	//classes bounding boxes
	vector<float> confidence;
	vector<int> classID;
	vector<Rect> boundingBox;
	
	//
	for(int i = 0; i < netOut.size(); i++)
	{
		float* data = (float*)netOut[i].data;
		for(int j = 0; j < netOut[i].rows; j++)
		{
			Mat scores = netOut[i].row(j).colRange(5, netOut[i].cols);
			Point classIdPoint;
			double maxVal;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &maxVal, 0, &classIdPoint);
			float objPrediction = data[4];
			if(objPrediction >= this->confThreshold)
			{
				int centerX = (int)(data[0] * cap.cols);
				int centerY = (int)(data[1] * cap.rows);
				int box_width = (int)(data[2] * cap.cols);
				int box_height = (int)(data[3] * cap.rows);
				
				confidence.push_back(maxVal);
				classID.push_back(classIdPoint.x);
				boundingBox.push_back(Rect(centerX, centerY, box_width, box_height));
			}
			data += netOut[i].cols;
		}
	}
	
	//remove bounding boxes that shows the similar object using Non-Maximum Suppression
	vector<int> indices;
	NMSBoxes(boundingBox, confidence, this->confThreshold, this->nmsThreshold, indices);
	
	//save the bounding boxes
	this->object.resize(indices.size());
	for(int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		yoloClass class_object_struct;
		class_object_struct.boundBox = boundingBox[idx];
		class_object_struct.classID = this->classes[classID[idx]];
		class_object_struct.confidence = confidence[idx];
		
		this->object[i] = class_object_struct;
	}
	
	return;
}
//return object function
vector<yoloClass> YoloNetwork::outputObject()
{
	return this->object;
}



/********************************************************************************************************************
* C++ Libraries
********************************************************************************************************************/
#include <iostream>	//C++ input/output stream library
#include <fstream>	//C++ file stream library

/********************************************************************************************************************
* OpenCV Libraries
********************************************************************************************************************/
#include <opencv2/dnn/dnn.hpp>	//OpenCV Deep Neural Network Library
#include <opencv2/imgproc.hpp>	//OpenCV Image Processing Library
#include <opencv2/highgui.hpp>	//OpenCV High-Level GUI Library
#include <opencv2/opencv.hpp>	//OpenCV Deep Neural Network Library

/********************************************************************************************************************
* Namespace scope ID - C++ and OpenCV
********************************************************************************************************************/
using namespace cv;	//OpenCV Computer Vision namespace scope ID
using namespace dnn;	//OpenCV Deep Neural Network namespace scope ID
using namespace std;	//C++ standard namespace scope ID

/********************************************************************************************************************
* Global Variable
********************************************************************************************************************/
const float SOCIAL_DISTANCE = 6;	//Constant Distance in feet (6ft)

/********************************************************************************************************************
* Struct yoloClass - stores the required data of each person in the frame
*	x 		- Stores x coordinate of the person in the frame	  
*	y 		- Stores y coordinate of the person in the frame
*	noncompliance	- Stores the person social distance flag
*			- true if the person is at or more than 6 feet. False if they are too close
********************************************************************************************************************/
struct yoloClass
{
	int x;
	int y;
	bool noncompliance;
};

/********************************************************************************************************************
* Object Detection Class Network - YoloNetwork
********************************************************************************************************************/
class YoloNetwork
{
	private:
		Net net;			//Artificial Neural Network declaration
		float confThreshold;		//Float type of Confidence threshold
		float nmsThreshold;		//Float type of Non-maximum suppression threshold
		int width, height;		//Int type of width and height of the frame
		vector<string> classes;		//Vector string to store classes names for coco.names
		vector<string> outputNames;	//Vector string to store classes names for output names
		vector<yoloClass> object;	//Vector type of yoloClass object variable
		int countPerson;		//Int type of counter for current people in the frame
		
	public:
		YoloNetwork(const string configFile, const string weightFile,
		const string coconames);					//Constructor
		~YoloNetwork();							//Destructor
		void CurrentFrame(Mat cap);					//Process current frame function
		vector<yoloClass> getOutputObject();				//Return vector type of yoloClass
		float Euclidean(struct yoloClass a, struct yoloClass b);	//Euclidean Distance Calculation
		int getNumberPeople();						//Return number of people in the frame
};

/********************************************************************************************************************
* Person Detection Class Constructor
********************************************************************************************************************/
YoloNetwork::YoloNetwork(const string configFile, const string weightFile, const string coconames)
{

	this->confThreshold = 0.6;	//Set confidence threshold for person detection to 0.5
	this->nmsThreshold = 0.6;	//Set non-maximum suppression threshold for person detection to 0.4
	this->width = 416;		//Set width of the input frame
	this->height = 416;		//Set height of the input frame
	
	/*
	   Open object detection class file to read the 80 classes names (coco.names)
	*/
	ifstream classes_ifs(coconames.c_str());
	string line;
	if(classes_ifs.is_open())
	{
		//get classes names and store into vector classes
		while(getline(classes_ifs, line))
		{
			this->classes.push_back(line);
		}
	}
	/*
	   Load in the network the YOLO configuration file and weight file by Joseph Redmon.
	   Ask the network to use CUDA computation when possible.
	*/
	this->net = readNetFromDarknet(configFile, weightFile);
	this->net.setPreferableBackend(DNN_BACKEND_CUDA);
	this->net.setPreferableTarget(DNN_TARGET_CUDA);
	
	//Get the last output layer names from YOLO
	this->outputNames = this->net.getUnconnectedOutLayersNames();
}

/********************************************************************************************************************
* Person Detection Class Destructor
********************************************************************************************************************/
YoloNetwork::~YoloNetwork()
{

}


/**********************************************************************
* Current Frame Processing Function
*********************************************************************/
void YoloNetwork::CurrentFrame(Mat cap)
{
	//convert the frame to a 4-D blob and pass to the YOLO object detector
	Mat blob;
	blobFromImage(cap, blob, 1/255.0, Size(this->width, this->height), Scalar(0,0,0), true, false);
	
	//set the network input
	net.setInput(blob);
	
	//running the network
	vector<Mat> netOut;
	net.forward(netOut, this->outputNames);
	
	vector<float> confidence;	//Vector to store confidence returned by the Object Detector
	vector<int> classID;		//Vector to store classID returned by the Object Detector
	vector<Rect> boundingBox;	//Vector to store the bounding boxes returned by the Object Detector
	
	//retrieve the output layers
	for(int i = 0; i < netOut.size(); i++)
	{
		/*
		  scan through the bounding boxed created by the Object Detector and 
		  retrieve only the the ones with the highest confidence score.
		  Set the class box label with the highest class confidence
		*/
		float* data = (float*)netOut[i].data;
		for(int j = 0; j < netOut[i].rows; j++)
		{
			Mat scores = netOut[i].row(j).colRange(5, netOut[i].cols);
			Point classIdPoint;
			double maxVal;
			
			//Get the value and location of the maximum score
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
	
	//save the highest confidence and person class location
	this->object.resize(indices.size());
	for(int i = 0; i < indices.size(); i++)
	{
		int idx = indices[i];
		if(this->classes[classID[idx]] == "person")
		{
			yoloClass class_object_struct;
			class_object_struct.x = boundingBox[idx].x;
			class_object_struct.y = boundingBox[idx].y;
			this->object[i] = class_object_struct;
			//this->countPerson++;
		}
	}

	/*
	  Traverse through each person in the current frame and calculate the distance
	  between each person and mark the flag as true if the distance between them are
	  not in compliance with the Social Distancing rule (6 feet)
	*/
	for(int x = 0; x < object.size(); x++)
	{
		for(int y = x+1; y < object.size(); y++)
		{
			//If more than one person in the frame
			if(object.size() != 1)
			{
				/*
				  Get the distance from the Euclidean function by passing two person data
				  to get the calculated distance between the coordinates of these two person
				*/
				float distance = Euclidean(object[x], object[y]);
				/*
				  if the distance is less than the distance threshold (6 feet),
				  mark these two person as true for not following the social
				  distancing threshold
				*/
				if(distance < SOCIAL_DISTANCE)
				{
					object[x].noncompliance = true;
					object[y].noncompliance = true;
				}
			}
		}
	}
	
	/*
	  this will be to communicate with the arduino
	*/
	//cout << "number of people: " << this->countPerson;
	//this->countPerson = 0;
	
	return;
}

/********************************************************************************************************************
* Euclidean Distance Function
********************************************************************************************************************/
float YoloNetwork::Euclidean(struct yoloClass a, struct yoloClass b)
{
    	/*
    	  Euclidean distance is used as the camera will be positioned to observe
    	  the room as a plane
    	  Get the difference between two points and square them for horizontal and vertical
    	  coordinates.
    	  Get the square root of the sum of the horizontal and vertical new values
    	  The euclidean is the distance in pixels of the frame
    	*/
   	float new_x = (b.x - a.x)^2;
    	float new_y = (b.y - a.y)^2;
    	float euclidean = sqrt(new_x + new_y);
    	//cout << "Euclidean: " << euclidean << endl;
	/*
	  The ratio is determined by obtaining the original distance during testing and
	  perform a division with the calculated euclidean.
	  ratio = known distance / euclidean in pixel
	*/
    	float ratio = 24.9166/23.0651; // /23.0651

	//The distance is scaled based on the ratio obtained
    	float distance = ratio * euclidean;
    	//cout << "distance: " << distance << endl;
    
	return distance;	//Return the distance
}

/********************************************************************************************************************
* Person Data In The Current Frame Function
********************************************************************************************************************/
vector<yoloClass> YoloNetwork::getOutputObject()
{
	/*
	  Return vector type yoloClass
	  Return the detected people in the current frame that has been processed
	*/
	return this->object;
}
/********************************************************************************************************************
* Number of People In The Current Frame Function
********************************************************************************************************************/
int YoloNetwork::getNumberPeople()
{
	/*
	  Return int type of current people in the frame
	*/
	return this->countPerson;
}


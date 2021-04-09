/*

Requirements:
	- OpenCV CUDA
	- CUDA
	- Yolov3 tiny weight file - yolov3-tiny.weights
	- Yolov3 tiny configuration file

To compile, paste in terminal:
	g++ main.cpp -o output `pkg-config --cflags --libs opencv`

*/


#include "Yolo.hpp"
//#include "Coords.hpp"

string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) + ", height=(int)" +
           to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + to_string(flip_method) + " ! video/x-raw, width=(int)" + to_string(display_width) + ", height=(int)" +
           to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

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
		if(this->classes[classID[idx]] == "person")
		{
			yoloClass class_object_struct;
			class_object_struct.x = boundingBox[idx].x;
			class_object_struct.y = boundingBox[idx].y;
			class_object_struct.classID = this->classes[classID[idx]];
			class_object_struct.confidence = confidence[idx];
			
			this->object[i] = class_object_struct;
			
		}
	}

	for(int x = 0; x < object.size(); x++)
	{
		for(int y = x+1; y < object.size(); y++)
		{
			float distance = euclidean(object[x], object[y]);
			cout << "distance2: " << distance << endl;
			if(distance < SOCIAL_DISTANCE)
			{
				object[x].flag = true;
			}
			
		}
	}
	
	return;
}

float YoloNetwork::Euclidean(struct yoloClass a, struct yoloClass b)
{
    	//distance between two points
   	float new_x = (b.x - a.x)^2;
    	float new_y = (b.y - a.y)^2;

    	float euclidean = sqrt(new_x + new_y);
	cout << "distance pixel: " << euclidean << endl;

    	float ratio = 3.916667/37.4299;

    	float distance = ratio * euclidean;
    

	//return distance;    
	return euclidean;
}

float YoloNetwork::EuclideanS(struct yoloClass a)
{
    	//distance between two points
   	float new_x = (a.x)^2;
    	float new_y = (a.y)^2;

    	//float euclidean = sqrt(new_x + new_y);
	float euclidean = sqrt(new_x + new_y);
	//cout << "distance pixel: " << euclidean << endl;

    	float ratio = 3.916667/37.4299;
	//cout << "ratio: " << ratio << endl;

    	float distance = ratio * euclidean;
	cout << "distance real: " << distance << endl;

	//return distance;    
	return distance;
}

//return object function
vector<yoloClass> YoloNetwork::outputObject()
{
	return this->object;
}

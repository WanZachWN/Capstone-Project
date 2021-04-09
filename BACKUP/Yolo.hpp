/********************************************************************************************************************
* Struct yoloClass - stores the data of each person in the frame
*	x - x coordinate of the person in the frame
*	y - y coordinate of the person in the frame
*	classID - class identification given by the Darknet Neural Network
*	confidence - the confidence of each 
********************************************************************************************************************/
static struct yoloClass
{
	int x;
	int y;
	float confidence;
	float distance;
	bool flag;
}yoloClass;


/********************************************************************************************************************
*
********************************************************************************************************************/
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
		float Euclidean(struct yoloClass a, struct yoloClass b);
		float EuclideanS(struct yoloClass a);
};




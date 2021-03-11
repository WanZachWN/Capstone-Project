#include <iostream>
#include <wiringPiI2C.h>

using namespace std;

#define DEVICE_ID 0x08

int main(int argc, argv)
{
	int fd = wiringPiI2CSetup(DEVICE_ID);
	if(fd == -1)
	{
		cout << "Failed to establish I2C connection" << endl;
		return -1;
	}

	cout << "communication success" << endl;

	uint8_t data = 10;
	wiringPiI2CWrite(fd, data);
	cout << "Sent: " << data << endl;

	return 0;
}
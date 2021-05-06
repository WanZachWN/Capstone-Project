from smbus import SMBus
 
addr = 0x08 # bus address
bus = SMBus(1) # indicates /dev/ic2-1
 
numb = 0
numbold = 0
string = ""

def isInt(value):
	try:
		int(value)
		return True
	except ValueError:
		return False

while string != "exit":
	f = open("number_people.txt", "r")
	f2 = open("compliance.txt", "r");
	string2 = f2.readline()
	string = f.readline()
	if(isInt(string) == True):
		numb = int(string);
		if isinstance(numb, int):
			if numb != numbold:
				numbold = numb
				numb2 = int(string2)
				if isinstance(numb2, int):
					if numb2 == 1:
						numb = numb + 64;
					bus.write_byte(addr, numb)
					print(numb)
	f.close()
	


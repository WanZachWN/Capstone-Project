EXECUTABLE := Yolo_1 \
		video_test \
		Yolo_test \

CODES := Yolo_1.cpp \
		video_test.cpp \
		Yolo_test \

.PHONY : all

.PHONY : clean

all :
	make $(EXECUTABLE)

clean :
	rm -f $(EXECUTABLE)

Yolo_test : Yolo_test.cpp
	g++ $^ -o $@ `pkg-config --cflags --libs opencv`

Yolo_1 : Yolo_1.cpp
	g++ $^ -o $@ `pkg-config --cflags --libs opencv`

video_test : video_test.cpp
	g++ $^ -o $@ `pkg-config --cflags --libs opencv`
 

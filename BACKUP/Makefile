OBJ := Yolo main \

SRC := Yolo.cpp main.cpp \

.PHONY : all

.PHONY : clean

all :
	make $(EXECUTABLE)

clean :
	rm -f $(EXECUTABLE)

Yolo : Yolo.cpp
	g++ $^ -o $@ `pkg-config --cflags --libs opencv`

main : main.cpp
	g++ $^ -o $@ `pkg-config --cflags --libs opencv`

 

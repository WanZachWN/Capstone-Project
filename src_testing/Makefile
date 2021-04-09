EXECUTABLE := main \

SRC := main.cpp \

.PHONY : all

.PHONY : clean

all :
	make $(EXECUTABLE)

clean :
	rm -f $(EXECUTABLE)

main : main.cpp
	g++ $^ -o $@ `pkg-config --cflags --libs opencv`

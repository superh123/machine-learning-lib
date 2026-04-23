CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O

SRCS = TestFile.cpp MLP.cpp Layer.cpp Node.cpp Matrix.cpp Activation.cpp

OBJS = $(SRCS:.cpp=.o)

TARGET = app

all: $(TARGET)

$(TARGET) : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
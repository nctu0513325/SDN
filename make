CC := g++
CFLAGS := -O3 -std=c++17 -Wall -pthread
TARGET := pi.out
OBJS := pi.o PPintrin.o logger.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

pi.o: pi.cpp PPintrin.h logger.h def.h
	$(CC) $(CFLAGS) -c pi.cpp

PPintrin.o: PPintrin.cpp PPintrin.h logger.h def.h
	$(CC) $(CFLAGS) -c PPintrin.cpp

logger.o: logger.cpp logger.h def.h
	$(CC) $(CFLAGS) -c logger.cpp

clean:
	rm -f $(TARGET) *.o

run:
	@if [ -z "$(THREADS)" ] || [ -z "$(TOSSES)" ]; then \
		echo "Usage: make run THREADS=<num> TOSSES=<num>"; \
	else \
		./$(TARGET) $(THREADS) $(TOSSES); \
	fi

.PHONY: all clean run

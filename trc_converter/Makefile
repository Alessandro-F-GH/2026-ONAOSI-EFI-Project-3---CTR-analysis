CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
ROOT_CFLAGS := $(shell root-config --cflags 2>/dev/null)
ROOT_LIBS := $(shell root-config --libs 2>/dev/null)

SRC := src/trc_to_root.cpp src/trc_reader.cpp src/file_discovery.cpp
OBJ := $(SRC:.cpp=.o)
TARGET := bin/trc_to_root

.PHONY: all clean check

all: check $(TARGET)

check:
	@command -v root-config >/dev/null 2>&1 || \
	  (echo "Error: root-config not found. Activate your ROOT environment first." >&2; exit 1)
	@echo "ROOT $$(root-config --version) from $$(root-config --prefix)"

$(TARGET): $(OBJ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) $(OBJ) $(ROOT_LIBS) -o $@

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(ROOT_CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

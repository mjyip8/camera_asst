HALIDE_BIN_PATH := $(HALIDE_PATH)/bin
HALIDE_INCLUDE_PATH := $(HALIDE_PATH)/include

SRC_DIR := src
BUILD_DIR := build
BIN_DIR := bin
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))
LDFLAGS := -L/usr/local/lib -lomp
CPPFLAGS := -std=c++11 -Wall -O2 -I/usr/local/include -Xpreprocessor -fopenmp
CXXFLAGS := -std=c++17

ifeq ($(USE_HALIDE), 1)
 CXXFLAGS += -D__USE_HALIDE__	-I$(HALIDE_INCLUDE_PATH)
 LDFLAGS += -L$(HALIDE_BIN_PATH) -lHalide -lpthread -ldl
endif

kcamera: $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	g++ -o $(BIN_DIR)/$@ $^ $(LDFLAGS)

clean:
	\rm -rf $(BUILD_DIR) $(BIN_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

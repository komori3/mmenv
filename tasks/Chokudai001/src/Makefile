LIB_PATH=
BUILD_PATH=build

CXXFLAGS+=-O2 -Wall -Wextra -Wno-sign-compare -std=c++17
CXXFLAGS+=-Wno-unused-variable -Wno-unused-function
CXXFLAGS+=-I.
CXXFLAGS+=-MMD -MP

# header-only libraries
CXXFLAGS+=-I../../../include

#CXXFLAGS+=-I~opencv/include/opencv2
#LDFLAGS+=-lopencv_core -lopencv_highgui -lopencv_imgproc

# add your cpp files here:
SRCS+=contest_types.cpp
OBJS=$(SRCS:%.cpp=$(BUILD_PATH)/%.o)
DEPS=$(SRCS:%.cpp=$(BUILD_PATH)/%.d)

# add your stable header files here to include them in the precompiled header:
PCH_HEADERS=
PCH_SRC=
PCH_OUT=

GEN_SRCS=$(SRCS) generator.cpp
GEN_OBJS=$(GEN_SRCS:%.cpp=$(BUILD_PATH)/%.o)
GEN_DEPS=$(GEN_SRCS:%.cpp=$(BUILD_PATH)/%.d)

JUDGE_SRCS=$(SRCS) judge.cpp
JUDGE_OBJS=$(JUDGE_SRCS:%.cpp=$(BUILD_PATH)/%.o)
JUDGE_DEPS=$(JUDGE_SRCS:%.cpp=$(BUILD_PATH)/%.d)

TARGETS=judge generator

.PHONY: all
all: dirs $(TARGETS)

.PHONY: clean
clean:
	rm -fr $(BUILD_PATH) $(TARGETS) $(PCH_OUT)

.PHONY: dirs
dirs:
	@mkdir -p $(BUILD_PATH)

$(PCH_OUT): $(PCH_SRC) $(PCH_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD_PATH)/%.o: %.cpp $(PCH_OUT)
	$(CXX) $(CXXFLAGS) -c $< -o $@

judge: $(JUDGE_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

generator: $(GEN_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

-include $(DEPS) $(JUDGE_DEPS)

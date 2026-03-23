#  Copyright 2026 Roman Kudinov. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


PROJECT = $(shell basename $(CURDIR))
SRC_DIR = src
TEST_DIR = test

SRC_ALL = $(wildcard src/*.cpp)
SRC_LIB = $(filter-out src/Tester.cpp, $(SRC_ALL))
SRC_TEST = $(wildcard test/*.cpp)

OBJ_LIB = $(SRC_LIB:$(SRC_DIR)/%.cpp=$(G_OBJ)/$(PROJECT)/%.o)
OBJ_TEST = $(SRC_TEST:$(TEST_DIR)/%.cpp=$(G_OBJ)/$(PROJECT)/%.o)

DEP_LIB = $(OBJ_LIB:%.o=%.d)
DEP_TEST = $(OBJ_TEST:%.o=%.d)

TEST_FILES = $(wildcard tester_files/*)
DST_TEST_FILES = $(TEST_FILES:tester_files/%=$(G_BIN)/$(PROJECT)/%)

INCLUDE_DIRS = -Iinclude -I$(G_SRC)/libcommon/include -I$(G_SRC)/liblogtrace/include $(INCLUDE_OPENCV) 
LIBRARY_DIRS = -L$(G_BIN)
CXXFLAGS = -fPIC -MMD $(INCLUDE_DIRS)
LDFLAGS = $(LIBRARY_DIRS) $(LDFLAGS_OPENCV) -lOpenCL -lcommon -llogtrace

.PHONY: all copy_test_files
all: $(G_BIN)/liblibocl.so $(G_BIN)/$(PROJECT)/test_libocl copy_test_files 
	@echo "\033[0;97m [DONE] $(PROJECT)"
	
copy_test_files : $(DST_TEST_FILES)
	
$(DST_TEST_FILES): $(G_BIN)/$(PROJECT)/% : tester_files/%
	@echo "\033[0;95m [COPY] $^ -> $@ \033[0;0m"
	@cp $^ $@
	
$(G_BIN)/libocl.so: $(OBJ_LIB)
	@echo "\033[0;32m [LINK] $@ \033[0;0m"
	g++ -shared -o $@ $^ $(LDFLAGS) $(LDFLAGS_OPENCV)
	
$(G_BIN)/$(PROJECT)/test_libocl: $(OBJ_TEST) $(G_BIN)/libocl.so
	@echo "\033[0;32m [LINK] $@ \033[0;0m"
	@g++ -o $@ $^ $(LDFLAGS) -lcommon -lgtest -lpthread

-include $(DEP_LIB)
-include $(DEP_TEST)

$(G_OBJ)/$(PROJECT)/%.o: $(SRC_DIR)/%.cpp
	@echo "\033[0;93m [CC] $@ \033[0;0m"
	@g++ -c $< -o $@ $(CXXFLAGS) $(G_CXXFLAGS) -std=c++14
	
$(G_OBJ)/$(PROJECT)/%.o: $(TEST_DIR)/%.cpp
	@echo "\033[0;93m [CC] $@ \033[0;0m"
	@g++ -c $< -o $@ $(CXXFLAGS) $(G_CXXFLAGS) -std=c++14

$(shell mkdir -p $(G_BIN)/$(PROJECT))
$(shell mkdir -p $(G_OBJ)/$(PROJECT))

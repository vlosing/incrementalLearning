################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/intf.cpp 

OBJS += \
./src/intf.o 

CPP_DEPS += \
./src/intf.d 


# Each subdirectory must supply rules for building sources it contributes
src/intf.o: ../src/intf.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/include/python2.7 -I/home/viktor/svn/viktlosi/Software/sources/OnlineRandomForest/eigen/install/include/eigen2 -I/usr/include/python2.7/numpy -I/home/viktor/svn/viktlosi/Software/sources/OnlineRandomForest/eclipseWorkspacePythonIntf/orfAdaptedToPythonIntf/src -O3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"src/intf.d" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



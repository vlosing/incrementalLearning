################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/grlvqtest.cpp 

OBJS += \
./src/grlvqtest.o 

CPP_DEPS += \
./src/grlvqtest.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I"/hri/localdisk/vlosing/OnlineLearning/eclipse_workspace_pub/grlvq_c" -I/hri//sit/latest/External/OpenCV/2.1/include/ -I"/hri/localdisk/vlosing/OnlineLearning/eclipse_workspace_pub/grlvq_c/src" -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



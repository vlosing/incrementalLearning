################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/intf.c 

OBJS += \
./src/intf.o 

C_DEPS += \
./src/intf.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -I/hri/sit/latest/External/AllPython/2.7/lucid64/include/python2.7 -I"/hri/localdisk/vlosing/OnlineLearning/eclipse_workspace_pub/grlvq_c/src" -I/hri/sit/latest/External/AllPython/2.7/lucid64/lib/numpy/core/include/numpy/ -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '



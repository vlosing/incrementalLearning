################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include src/subdir.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: grlvqtest_c

dependents:
	-cd /hri/localdisk/vlosing/OnlineLearning/eclipse_workspace_pub/grlvq_c/Debug && $(MAKE) clean all

# Tool invocations
grlvqtest_c: $(OBJS) $(USER_OBJS) /hri/localdisk/vlosing/OnlineLearning/eclipse_workspace_pub/grlvq_c/Debug/libgrlvqc.so
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++ -L"/hri/localdisk/vlosing/OnlineLearning/eclipse_workspace_pub/grlvq_c/Debug" -L/hri/sit/latest/External/OpenCV/2.1/lib/lucid32 -o "grlvqtest_c" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) grlvqtest_c
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:
/hri/localdisk/vlosing/OnlineLearning/eclipse_workspace_pub/grlvq_c/Debug/libgrlvqc.so:

-include ../makefile.targets

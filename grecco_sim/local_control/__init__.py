"""Local controllers for grid-connected systems.

This module provides various local controller implementations that extend LocalControllerBase:

Classes:
    LocalControllerSelfSuff: Implements self-sufficiency optimization without 
        considering grid constraints
    LocalControllerEVBaseline: Basic controller for EV charging that charges at maximum 
        power when connected
    LocalControllerHeatPumpOnOff: Controls heat pump operation based on temperature bounds 
        with on/off control
    LocalControllerNoControl: Simple pass-through controller for systems without controllable
        loads that always returns 0.0
    LocalControllerPassControl: Directly passes through control signals from central
        coordinator without modification
    LocalControllerADMM: Implements local optimization for the ADMM coordination scheme
    LocalControllerFirstOrder: Implements local optimization for first-order coordination
        methods like gradient descent

Each controller provides specific control logic for different types of systems or control
strategies in the grid simulation framework."
This means that for each coordination mechanism every controllable load has one respective
 controller.
"""

from grecco_sim.sim_data_tools import simbench_to_pypsa_conversion
import simbench

# Compare the two networks 
def comparison_simbench_pypsa(net_simbench, simbench_instance):

    if len(net_simbench.bus) == len(simbench_instance.net_pypsa.buses):
        print("Number of buses is equal")
    else:
        raise ValueError("Number of buses is unequal")

    if len(net_simbench.line) == len(simbench_instance.net_pypsa.lines):
        print("Number of lines is equal")
    else:
        raise ValueError("Number of lines is unequal")

    if (len(net_simbench.sgen) + len(simbench_instance.net_simbench.gen)) == len(simbench_instance.net_pypsa.generators):
        print("Number of generators is equal")
    else:
        raise ValueError("Number of generators is unequal")

    if len(net_simbench.load) == len(simbench_instance.net_pypsa.loads):
        print("Number of loads is equal")
    else:
        raise ValueError("Number of loads is unequal")

    if len(net_simbench.trafo) == len(simbench_instance.net_pypsa.transformers):
        print("Number of transformers is equal")
    else:
        raise ValueError("Number of transformers is unequal")

    if len(net_simbench.storage) == len(simbench_instance.net_pypsa.storage_units):
        print("Number of storage units is equal")
    else:
        raise ValueError("Number of storage units is unequal")

if __name__ == "__main__":
    # Import the SimBench network 
    sb_code = "1-LV-semiurb4--2-no_sw" # This network must match the one utilized to construct the PyPSA network
    net_simbench = simbench.get_simbench_net(sb_code)

    # Call the created function 
    simbench_instance  = simbench_to_pypsa_conversion.SimbenchToPypsa()
    net_pypsa = simbench_instance.simbench_to_pypsa()
    comparison_simbench_pypsa(net_simbench, simbench_instance)

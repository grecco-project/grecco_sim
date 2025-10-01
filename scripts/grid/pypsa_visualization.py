import pathlib
import pypsa 
import matplotlib.pyplot as plt
import cartopy.crs as crs
 
def visualize_pypsa_net(network):
    # Static plot 
    network.plot(title = "Static Plot of the Power Grid", margin = 0.1, 
                 bus_sizes = 0.3*1e-8)
    plt.show()

    # Interactive plots
    network.iplot(title = "Interactive Plot of the Power Grid",
                mapbox = True, mapbox_style = "open-street-map",
                # mapbox_style = "carto-positron"
                mapbox_parameters = {"zoom": 15})
    plt.show()

    # Bus voltage plot 
    fig, ax = plt.subplots(1, 1, subplot_kw = {"projection":crs.PlateCarree()})
    fig.set_size_inches(6, 6)
    network.plot(ax = ax, bus_sizes = 0.2*1e-10, title = "Bus Voltage")
    plt.hexbin(network.buses.x, network.buses.y, gridsize = 20, 
            C = network.buses_t.v_mag_pu.loc[network.snapshots[0]],
            cmap = plt.cm.jet) 
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    # Create empty PyPSA network 
    network = pypsa.Network()

    # Set the path 
    path = pathlib.Path(__file__).parent.absolute()
    path = path.parent.parent
    path = path / "results" / "power_flow_results"

    # Import the network's components 
    network.import_from_csv_folder(path)

    # Create different plots of the network 
    visualize_pypsa_net(network)
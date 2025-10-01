import pathlib

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from shapely.geometry import MultiPoint
from tespy.components import Compressor, CycleCloser, SimpleHeatExchanger, Valve
from tespy.connections import Connection
from tespy.networks import Network



class HeatPumpParameters:
    def __init__(self, p_max, cop, refrigerant, p_cut_off, slope, intercept):
        self.p_max = p_max
        self.cop = cop
        self.refrigerant = refrigerant
        self.p_cut_off = p_cut_off
        self.slope = slope
        self.intercept = intercept


class HeatPumpCreator():
    """
    This model class calculates the heat pump curve(pel vs cop)  during isobaric operation.  The selection of operational point,
    and structure of the thermodynamic calculation is defined in the methods

    Parameters: heat_pump_type - select from database.
                To each type, refrigerant, pressure levels and compressor power limits are defined
    """

    def __init__(self, parameters):

        # Input parameters
        self.p_max = parameters.p_max * 1000  # Transforming to W, necessary for using tepsy
        self.cop = parameters.cop
        self.refrigerant = parameters.refrigerant
        self.p_cut_off = parameters.p_cut_off
        self.slope = parameters.slope
        self.intercept = parameters.intercept

        # Attributes for operation of heat pump
        self.operational_data = None
        self.selected_point = None

    def get_operational_data(self):
        """
        Determine operational points (combination pressure in cold side and hot side).

        Return: dataframe with pressures, efficiency, mass of refrigerant, and temperature levels
        """
        my_plant = Network()
        my_plant.set_attr(t_unit='C', p_unit='bar', h_unit='kJ / kg')
        cc = CycleCloser('cycle closer')
        cond = SimpleHeatExchanger('condenser')
        evap = SimpleHeatExchanger('evaporator')
        va = Valve('expansion valve')
        comp = Compressor('compressor')

        # Create empty lists to store the results
        pressure_list_cold_values = []
        pressure_list_hot_values = []
        eta_values = []
        mass_values = []
        temp1_values = []
        temp4_values = []

        # connections of heat pump
        c1 = Connection(cc, 'out1', evap, 'in1', label='1')
        c2 = Connection(evap, 'out1', comp, 'in1', label='2')
        c3 = Connection(comp, 'out1', cond, 'in1', label='3')
        c4 = Connection(cond, 'out1', va, 'in1', label='4')
        c0 = Connection(va, 'out1', cc, 'in1', label='0')

        my_plant.add_conns(c1, c2, c3, c4, c0,)

        # Internal dictionary for operating pressures (hot and cold sides of the heat pump) 
        # Rounded values used for the lists 
        refrigerant_pressure_dict = {"R32":{"cold_side_pressure": list(range(2, 7)), "hot_side_pressure": list(range(19, 56))},
                                     "R290": {"cold_side_pressure": list(range(1, 4)), "hot_side_pressure": list(range(11, 31))},
                                     "R407c": {"cold_side_pressure": list(range(1, 4)), "hot_side_pressure": list(range(14, 42))}}

        pressure_list_cold = refrigerant_pressure_dict[self.refrigerant]["cold_side_pressure"]
        pressure_list_hot = refrigerant_pressure_dict[self.refrigerant]["hot_side_pressure"]

        for coldpress in pressure_list_cold:
            for hotpress in pressure_list_hot:
                # Import the DAIKIN speks
                # cop=2,7
                # Power consumption=1850 kW
                # Nomincal capacity for heating and cooling= 5kw
                comp.set_attr(P=self.p_max)
                cond.set_attr(pr=0.99, Q=-self.p_max * self.cop)  # pressure drop at the condenser (consumer side) with -10^6 Kj/kg heating need.
                evap.set_attr(pr=0.99)
                c2.set_attr(p=coldpress, x=1, fluid={self.refrigerant: 1})  # TODO Refrigerant mixture should be specified for other heat pumps
                c4.set_attr(p=hotpress, x=0)
                my_plant.solve(mode='design')
                pressure_list_cold_values.append(coldpress)
                pressure_list_hot_values.append(hotpress)
                eta_values.append(comp.eta_s.val)
                mass_values.append(c4.m.val)
                temp1_values.append(c1.T.val)
                temp4_values.append(c4.T.val)

        data = {
            'pressurelistcold': pressure_list_cold_values,
            'pressurelisthot': pressure_list_hot_values,
            'eta': eta_values,
            'mass': mass_values,
            'Temp1': temp1_values,
            'Temp4': temp4_values
        }

        df = pd.DataFrame(data)
        self.operational_data = df[(df['eta'] <= 0.95) & (df['Temp4'] - df['Temp1'] <= 70)]  # TODO: Formalize exlusion criteria
        self.operational_data["DeltaT"] = self.operational_data["Temp4"] - self.operational_data["Temp1"]

        return self.operational_data

    def find_operational_point(self):
        """
        Determine one single set of operating pressures

        Parameters:  Dataframe containing operational data from initiate_heat_pump()
        Return: dataframe with single operational point of pressures, efficiency, mass of refrigerant, and temperature levels
        """
        self.get_operational_data()  # Initializing self.operational_data
        points = self.operational_data[['mass', 'eta']].values.tolist()
        centroid = MultiPoint(points).centroid
        df_copy = self.operational_data.copy()
        df_copy.loc[:, 'distance_to_centroid'] = df_copy.apply(lambda row: euclidean((centroid.x, centroid.y), (row['mass'], row['eta'])), axis=1)
        self.selected_point = df_copy.loc[df_copy['distance_to_centroid'].idxmin()]

        return self.selected_point

    def get_heat_from_heatpump(self, p_in):
        """
        Get the heat output from the power input to the compressor

        Parameters:  p_in -  Power to the compressor
        Return: q_out - Heat to/into household
        """
        my_plant = Network()
        my_plant.set_attr(T_unit='C', p_unit='bar', h_unit='kJ / kg')
        cc = CycleCloser('cycle closer')
        cond = SimpleHeatExchanger('condenser')
        evap = SimpleHeatExchanger('evaporator')
        va = Valve('expansion valve')
        comp = Compressor('compressor')
        c1 = Connection(cc, 'out1', evap, 'in1', label='1')
        c2 = Connection(evap, 'out1', comp, 'in1', label='2')
        c3 = Connection(comp, 'out1', cond, 'in1', label='3')
        c4 = Connection(cond, 'out1', va, 'in1', label='4')
        c0 = Connection(va, 'out1', cc, 'in1', label='0')
        my_plant.add_conns(c1, c2, c3, c4, c0,)
        comp.set_attr(P=p_in)
        cond.set_attr(pr=0.99)
        evap.set_attr(pr=0.99)
        c2.set_attr(m=self.selected_point["mass"], x=1, fluid={self.refrigerant: 1})  # TODO Refrigerant mixture should be specified for other heat pumps
        c4.set_attr(x=0, p=self.selected_point["pressurelisthot"])                   # Perhaps is not relevant to have mixtures
        c1.set_attr(p=self.selected_point["pressurelistcold"])
        my_plant.solve(mode='design')

        # apply the cut off operation
        if comp.eta_s.val < 1:
            # cop=abs(cond.q.val) / comp.p.val
            q_out = abs(cond.Q.val)
        else:
            # cop = 0
            q_out = 0

        return q_out

    def generate_operation_curve(self):
        """
        Get line regression parameters of Heat generation vs Electricity consumption

        Parameters:  selected_point - operational point of pressures
                     p_max_of_hp : Maximum rated power of the heat pump
        Return: p_cut_off - Minimum input power to heat pump
                slope
                intercept
        """
        self.find_operational_point()  # Initializing self.selected_point before using it
        given_range = int(self.p_max)  # Watts
        qcond_list = []
        pel_value_list = []

        for pel_value in range(0, given_range + 1, 5):

            qcondenser = self.get_heat_from_heatpump(pel_value)
            qcond_list.append(qcondenser)
            pel_value_list.append(pel_value)

        qcond_series = pd.Series(qcond_list)
        pel_series = pd.Series(pel_value_list)
        non_zero_indices = qcond_series[qcond_series != 0].index
        # qcut_off=qcond_series[non_zero_indices[0]]
        self.p_cut_off = pel_series[non_zero_indices[0]] / 1000  # To kW
        cut_off_index = non_zero_indices[0]

        qcond_series_cut_off = qcond_series[cut_off_index:]
        pel_series_cut_off = pel_series[cut_off_index:]

        # Convert index and values to arrays
        x = pel_series_cut_off.values
        y = qcond_series_cut_off.values

        # Calculate the regression coefficients (slope and intercept) using numpy's polyfit function
        coefficients = np.polyfit(x, y, 1)

        # Extract the slope (a) and intercept (b) from the coefficients
        self.slope = coefficients[0]
        self.intercept = coefficients[1] / 1000  # transforming to kW

        return self.p_cut_off, self.slope, self.intercept

def identify_heatpump_model():
    # Import the heat pump database 
    path = pathlib.Path(__file__).parent.absolute()  # model_identification directory
    path = path.parent.parent.parent  # repo base directory
    path = path / "data" / "heat_pump_database" / "heat_pump_database_short_version.csv"
    heat_pump_info = pd.read_csv(path, sep=";")

    for i in heat_pump_info.index:
        p_max = heat_pump_info["p_max"].iloc[i]
        cop = heat_pump_info["cop"].iloc[i]
        refrigerant = heat_pump_info["refrigerant"].iloc[i]
        p_cut_off = heat_pump_info["p_cut_off"].iloc[i]
        slope = heat_pump_info["slope"].iloc[i]
        intercept = heat_pump_info["intercept"].iloc[i]

        parameters = HeatPumpParameters(p_max, cop, refrigerant, p_cut_off, slope, intercept)
        heat_pump_instance = HeatPumpCreator(parameters)
        if str(parameters.slope) == "nan":
            print(f"Generating parameters for heat pump {i}")
            heat_pump_instance.generate_operation_curve()  
        else:
            print(f"Information for heat pump {i} was already available. No calculation was performed.")
            
        heat_pump_info.at[i, "p_cut_off"] = heat_pump_instance.p_cut_off
        heat_pump_info.at[i, "slope"] = heat_pump_instance.slope
        heat_pump_info.at[i, "intercept"] = heat_pump_instance.intercept

    # Export data to the heat pump database  
    heat_pump_info.to_csv(path, sep=";", index = False)



if __name__ == "__main__":
    identify_heatpump_model()
    
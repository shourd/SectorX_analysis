"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

from data_objects import logpointAircraft, scenarioAircraft
import math


def get_relevant_aircraft(relevant_aircraft_list, scenario_aircraft, finished_aircraft):

    # Loop through each aircraft in the logpoint
    for i_aircraft, aircraft in enumerate(relevant_aircraft_list[:]):

        # Check if aircraft has already finished using the ACID in the finished aircraft list
        if aircraft.ACID in finished_aircraft[:]:

            # Delete from relevant aircraft list      
            for iRelevantAircraft, relevantAircraft in enumerate(relevant_aircraft_list[:]):
                if relevantAircraft.ACID == aircraft.ACID:
                    relevant_aircraft_list.remove(relevantAircraft)

        else:

            # Loop through each scenario aircraft to get the COPX positions
            iSAircraft: int = None
            sAircraft: scenarioAircraft = None
            for iSAircraft, sAircraft in enumerate(scenario_aircraft[:]):

                # If the logpoint aircraft is the same as the scenario aircraft
                if aircraft.ACID == sAircraft.ACID:

                    # Delta distances between aircraft and its COPX [nm]
                    dx = sAircraft.COPX_x_nm - aircraft.x_nm
                    dy = sAircraft.COPX_y_nm - aircraft.y_nm
                    dxy = math.sqrt(dx*dx + dy*dy)

                    # Check if aircraft reached its goal and a TOC has been issued in this logpoint
                    if dxy < 5.0 and aircraft.toc:

                        # Remove aircraft from list
                        for iRelevantAircraft, relevantAircraft in enumerate(relevant_aircraft_list[:]):
                            if relevantAircraft.ACID == aircraft.ACID:
                                relevant_aircraft_list.remove(relevantAircraft)

                        # Add aircraft to finished aircraft list
                        finished_aircraft.append(aircraft.ACID)

    return relevant_aircraft_list, finished_aircraft

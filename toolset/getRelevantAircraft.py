"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

from dataObjects import logpointAircraft, scenarioAircraft
import math


def getRelevantAircraft(_logpointAircraft, _scenarioAircraft, _finishedAircraft):

    # Relevant aircraft list
    relevantAircraftList = _logpointAircraft

    # Finished aircraft list
    finishedAircraft = _finishedAircraft

    # Loop through each aircraft in the logpoint
    iAircraft: int = None
    aircraft: logpointAircraft = None
    for iAircraft, aircraft in enumerate(_logpointAircraft[:]):

        # Check if aircraft has already finished using the ACID in the finished aircraft list
        if aircraft.ACID in finishedAircraft[:]:

            # Delete from relevant aircraft list      
            for iRelevantAircraft, relevantAircraft in enumerate(relevantAircraftList[:]):
                if relevantAircraft.ACID == aircraft.ACID:
                    relevantAircraftList.remove(relevantAircraft)

        else:

            # Loop through each scenario aircraft to get the COPX positions
            iSAircraft: int = None
            sAircraft: scenarioAircraft = None
            for iSAircraft, sAircraft in enumerate(_scenarioAircraft[:]):

                # If the logpoint aircraft is the same as the scenario aircraft
                if aircraft.ACID == sAircraft.ACID:

                    # Delta distances between aircraft and its COPX [nm]
                    dx = sAircraft.COPX_x_nm - aircraft.x_nm
                    dy = sAircraft.COPX_y_nm - aircraft.y_nm
                    dxy = math.sqrt(dx*dx + dy*dy)

                    # Check if aircraft reached its goal and a TOC has been issued in this logpoint
                    if dxy < 5.0 and aircraft.toc:

                        # Remove aircraft from list
                        for iRelevantAircraft, relevantAircraft in enumerate(relevantAircraftList[:]):
                            if relevantAircraft.ACID == aircraft.ACID:
                                relevantAircraftList.remove(relevantAircraft)

                        # Add aircraft to finished aircraft list
                        finishedAircraft.append(aircraft.ACID)

    return relevantAircraftList, finishedAircraft
from dataObjects import logpointAircraft
from tools import ISA_IAStoTAS_kts, ftToNm, nmToFt
import math
import numpy
from shapely import geometry


def getConflicts (aircraftList, cautionTime, warningTime, accountForTOC, sectorPoints, distanceToSector):

    # Conflict list
    conflictList = []

    # Create sector polygon
    sectorPolygon = geometry.Polygon(sectorPoints)

    # Loop through each aircraft
    iAircraft: int = None
    aircraft: logpointAircraft = None
    for iAircraft, aircraft in enumerate(aircraftList):
        
        # Get true airspeed
        aircraftTAS = ISA_IAStoTAS_kts(aircraft.spd_kts, aircraft.alt_ft)

        # Aircraft position and velocity vector
        aircraftPos = [aircraft.x_nm, aircraft.y_nm]
        aircraftVelocity = [aircraftTAS * math.sin(math.radians(aircraft.hdg_deg)),
                            aircraftTAS * math.cos(math.radians(aircraft.hdg_deg))]

        # Loop through each relative aircraft
        iRelativeAircraft: int = None
        relativeAircraft: logpointAircraft = None
        for iRelativeAircraft, relativeAircraft in enumerate(aircraftList):

            # If the relative aircraft is the same as the current aircraft, then skip to the next relative aircraft
            if (aircraft.ACID == relativeAircraft.ACID):
                continue

            # Get true airspeed
            relativeAircraftTAS = ISA_IAStoTAS_kts(relativeAircraft.spd_kts, relativeAircraft.alt_ft)
            
            # Relative aircraft position and velocity vector
            relativeAircraftPos = [relativeAircraft.x_nm, relativeAircraft.y_nm]
            relativeAircraftVelocity = [relativeAircraftTAS * math.sin(math.radians(relativeAircraft.hdg_deg)),
                                        relativeAircraftTAS * math.cos(math.radians(relativeAircraft.hdg_deg))]

            # Delta distance [nm]
            dx = relativeAircraftPos[0] - aircraftPos[0]
            dy = relativeAircraftPos[1] - aircraftPos[1]

            # Delta speed [nm/h]
            dvx = relativeAircraftVelocity[0] - aircraftVelocity[0]
            dvy = relativeAircraftVelocity[1] - aircraftVelocity[1]

            # Calculate 2D time to CPA [h]
            if( not(dvx == 0.0 and dvy == 0.0) ):
                t_CPA = -(dx * dvx + dy * dvy) / (dvx * dvx + dvy * dvy)
            else:
                t_CPA = 0.0
            
            # Calculate 2D distance to CPA [nm]
            d_CPA = math.sqrt((dx + dvx*t_CPA)*(dx + dvx*t_CPA) + (dy + dvy*t_CPA)*(dy + dvy*t_CPA))

            # Calculate time to loss of separation [h]
            if( not(dvx == 0.0 and dvy == 0.0) and d_CPA < 5 ):
                t_LOS = t_CPA - math.sqrt(5.0*5.0 - d_CPA*d_CPA) / math.sqrt(dvx*dvx + dvy*dvy)
            else:
                t_LOS = 0.0

            # Aircraft position at CPA
            aircraftPOSAtCPA = [aircraftPos[0] + aircraftVelocity[0] * t_CPA,
                                aircraftPos[1] + aircraftVelocity[1] * t_CPA]

            # Relative aircraft position at CPA
            relativeAircraftPOSAtCPA = [relativeAircraftPos[0] + relativeAircraftVelocity[0] * t_CPA,
                                        relativeAircraftPos[1] + relativeAircraftVelocity[1] * t_CPA]

            #print(str(t_LOS > 0) + ":" +
            #      str(t_LOS < cautionTime / 3600) + ":" +
            #      str(numpy.round(numpy.abs(relativeAircraft.alt_ft - aircraft.alt_ft)) < 1000) + ":" +
            #      str(math.hypot(relativeAircraftPOSAtCPA[0] - aircraftPOSAtCPA[0], relativeAircraftPOSAtCPA[1] - aircraftPOSAtCPA[1]) < 5) + ":" +
            #      str(sectorPolygon.distance(geometry.Point(aircraftPos)) < distanceToSector) + ":" +
            #      str(not (accountForTOC and aircraft.toc and relativeAircraft.toc)))

            # Check if aircraft will be in conflict
            if (# Positive time to LOS
                #t_LOS > 0 and
                # Time to LOS is less than the caution time
                t_LOS < cautionTime / 3600 and
                # Will have vertical LOS
                numpy.round(numpy.abs(relativeAircraft.alt_ft - aircraft.alt_ft)) < 1000 and 
                # Will have horizontal LOS
                math.hypot(relativeAircraftPOSAtCPA[0] - aircraftPOSAtCPA[0], relativeAircraftPOSAtCPA[1] - aircraftPOSAtCPA[1]) < 5 and
                # Check if aircraft is not too far from the sector
                sectorPolygon.distance(geometry.Point(aircraftPos)) < distanceToSector and
                # Check if a TOC has been issued
                not (accountForTOC and aircraft.toc and relativeAircraft.toc)):
                    willBeInConflict = True
            else:
                    willBeInConflict = False

            # If the aircraft will be in conflict
            if (willBeInConflict):

                # Get conflict angle [rad]
                conflictAngle = numpy.arccos(numpy.clip(numpy.dot(aircraftVelocity/numpy.linalg.norm(aircraftVelocity),relativeAircraftVelocity/numpy.linalg.norm(relativeAircraftVelocity)),-1.0,1.0))

                # If conflict list is empty, then store the conflict
                if not (conflictList):
                    alreadyExists = False
                else:

                    # Check if conflict pair is already in the list
                    alreadyExists = False
                    for i in range(0, len(conflictList)):
                        if ((conflictList[i][0] == aircraft.ACID and conflictList[i][1] == relativeAircraft.ACID) or 
                           (conflictList[i][1] == aircraft.ACID and conflictList[i][0] == relativeAircraft.ACID)):
                            alreadyExists = True
                            
                # If conflct pair is not in the list, then store the conflict
                if not (alreadyExists):
                    conflictList.append([aircraft.ACID, relativeAircraft.ACID, t_CPA*3600, d_CPA, t_LOS*3600, math.degrees(conflictAngle)])
    
    # Return the conflict list                    
    return conflictList

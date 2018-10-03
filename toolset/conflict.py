from data_objects import logpointAircraft
from tools import ISA_IAStoTAS_kts, ftToNm, nmToFt
import math
import numpy
from shapely import geometry
from config import Settings


def get_conflicts(aircraft_list, sector_points, settings):
    account_for_TOC = False
    # Conflict list
    conflict_list = []

    # Create sector polygon
    sector_polygon = geometry.Polygon(sector_points)

    # Loop through each aircraft
    for i_aircraft, aircraft in enumerate(aircraft_list):
        
        # Get true airspeed
        aircraft_TAS = ISA_IAStoTAS_kts(aircraft.spd_kts, aircraft.alt_ft)

        # Aircraft position and velocity vector
        aircraft_pos = [aircraft.x_nm, aircraft.y_nm]
        aircraft_velocity = [aircraft_TAS * math.sin(math.radians(aircraft.hdg_deg)),
                             aircraft_TAS * math.cos(math.radians(aircraft.hdg_deg))]

        # Loop through each relative aircraft
        for i_relative_aircraft, relative_aircraft in enumerate(aircraft_list):

            # If the relative aircraft is the same as the current aircraft, then skip to the next relative aircraft
            if aircraft.ACID == relative_aircraft.ACID:
                continue

            # Get true airspeed
            relative_aircraft_tas = ISA_IAStoTAS_kts(relative_aircraft.spd_kts, relative_aircraft.alt_ft)
            
            # Relative aircraft position and velocity vector
            relative_aircraft_pos = [relative_aircraft.x_nm, relative_aircraft.y_nm]
            relative_aircraft_velocity = [relative_aircraft_tas * math.sin(math.radians(relative_aircraft.hdg_deg)),
                                          relative_aircraft_tas * math.cos(math.radians(relative_aircraft.hdg_deg))]

            # Delta distance [nm]
            dx = relative_aircraft_pos[0] - aircraft_pos[0]
            dy = relative_aircraft_pos[1] - aircraft_pos[1]

            # Delta speed [nm/h]
            dvx = relative_aircraft_velocity[0] - aircraft_velocity[0]
            dvy = relative_aircraft_velocity[1] - aircraft_velocity[1]

            # Calculate 2D time to CPA [h]
            if not(dvx == 0.0 and dvy == 0.0):
                t_CPA = -(dx * dvx + dy * dvy) / (dvx * dvx + dvy * dvy)
            else:
                t_CPA = 0.0
            
            # Calculate 2D distance to CPA [nm]
            d_CPA = math.sqrt((dx + dvx*t_CPA)*(dx + dvx*t_CPA) + (dy + dvy*t_CPA)*(dy + dvy*t_CPA))

            # Calculate time to loss of separation [h]
            if not(dvx == 0.0 and dvy == 0.0) and d_CPA < 5:
                t_LOS = t_CPA - math.sqrt(5.0*5.0 - d_CPA*d_CPA) / math.sqrt(dvx*dvx + dvy*dvy)
            else:
                t_LOS = 0.0

            # Aircraft position at CPA
            aircraft_pos_at_CPA = [aircraft_pos[0] + aircraft_velocity[0] * t_CPA,
                                   aircraft_pos[1] + aircraft_velocity[1] * t_CPA]

            # Relative aircraft position at CPA
            relative_aircraft_pos_at_CPA = [relative_aircraft_pos[0] + relative_aircraft_velocity[0] * t_CPA,
                                            relative_aircraft_pos[1] + relative_aircraft_velocity[1] * t_CPA]

            # Check if aircraft will be in conflict
            if (  # Positive time to LOS
                t_CPA > 0 and
                # Time to LOS is less than the caution time
                t_LOS < settings.caution_time / 3600 and
                # Will have vertical LOS
                numpy.round(numpy.abs(relative_aircraft.alt_ft - aircraft.alt_ft)) < 1000 and
                # Will have horizontal LOS
                math.hypot(relative_aircraft_pos_at_CPA[0] - aircraft_pos_at_CPA[0], relative_aircraft_pos_at_CPA[1] - aircraft_pos_at_CPA[1]) < 5 and
                # Check if aircraft is not too far from the sector
                sector_polygon.distance(geometry.Point(aircraft_pos)) < settings.distance_to_sector and
                # Check if a TOC has been issued
                not(account_for_TOC and aircraft.toc and relative_aircraft.toc)):
                    in_conflict = True
            else:
                    in_conflict = False

            # If the aircraft will be in conflict
            if in_conflict:

                # Get conflict angle [rad]
                conflict_angle = numpy.arccos(numpy.clip(numpy.dot(aircraft_velocity/numpy.linalg.norm(aircraft_velocity),
                                                                   relative_aircraft_velocity/numpy.linalg.norm(relative_aircraft_velocity)),-1.0,1.0))

                # If conflict list is empty, then store the conflict
                if not conflict_list:
                    already_exists = False
                else:

                    # Check if conflict pair is already in the list
                    already_exists = False
                    for i in range(0, len(conflict_list)):
                        if ((conflict_list[i][0] == aircraft.ACID and conflict_list[i][1] == relative_aircraft.ACID) or
                           (conflict_list[i][1] == aircraft.ACID and conflict_list[i][0] == relative_aircraft.ACID)):
                            already_exists = True
                            
                # If conflict pair is not in the list, then store the conflict
                if not already_exists:
                    conflict_list.append([aircraft.ACID, relative_aircraft.ACID, t_CPA*3600, d_CPA, t_LOS*3600, math.degrees(conflict_angle)])
    
    # Return the conflict list                    
    return conflict_list


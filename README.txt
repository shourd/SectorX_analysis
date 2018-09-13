----------------------------------
-- SECTORX XML LOGDATA ANALYSIS --
----------------------------------

SectorX is an Air Traffic Control Simulator developed at TUDelft (C. Borst).
These python scripts analyze and visualize the generated data.

-- Original files created by T. de Jong --

serializeData.py serializes data using dataObjects.py.

get_relevant_aircraft.py: provides a list of all a/c that have to be taken into consideration: out of sector a/c are removed

conflict.py: provides a list of all a/c in conflict per logpoint with tCPA, dCPA, tLOS, conflict angle.

-- DataFrames and plotting --

processData.py: converts data to dataframes and plot


----------------------------------
-- SECTORX XML LOGDATA ANALYSIS --
----------------------------------

-- Original files created by Tjitte de Jong --

Readme in Dutch:
getDataFiles.py verzamelt alle msg en xml bestanden in 1 grote map.

serializeData.py serializet de data aan de hand van de structuur in dataObjects.py zodat de data sneller verwerkt kan worden.

processData3.py verwerkt de geserialiseerde data (dit bestand is nog lang niet af).

getRelevantAircraft.py geeft per logpunt een lijst met de relevante vliegtuigen die nodig zijn voor de verwerking
(vliegtuigen die hun doel gehaald hebben of nog niet in de sector zijn staan niet in de lijst).

conflict.py geeft per logpunt een lijst met vliegtuigen die in conflict zijn (met o.a. tCPA, dCPA, tLOS, conflict angle).

tools.py bevat kleine handige functies zoals unit conversion.
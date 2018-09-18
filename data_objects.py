"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

from tools import strToBool


class Participant:
    # Initialization function
    def __init__(self, name):
        self.name = name
        self.runs = []  # create empty list


# class Run:
#     # Initialization function
#     def __init__(self, participant, record, commands, file_name):
#         self.participant = participant
#         self.record = record
#         self.commands = commands
#         self.file_name = file_name


class Command:
    # Initialization function
    def __init__(self, commandLine):

        # Variables default values
        self.timestamp = None
        self.ERROR = None
        self.ACID = None
        self.EFL = None
        self.HDG = None
        self.SPD = None
        self.DCT = False
        self.TOC = False
        self.SVE = False
        self.PZN = False
        self.CLR = False
        self.PRV = False
        self.EXQ = False

        # Split the command line into the individual commands and values
        commandLine = commandLine.split(":")

        # Get timestamp
        self.timestamp = float(commandLine[0])

        # If there is an error
        if commandLine[1] == "ERROR":
            self.ERROR = commandLine[2]

        # Else if speed vector on/off
        elif commandLine[1] == "SVE":
            self.SVE = True

        # Else if protected zone on/off
        elif commandLine[1] == "PZN":
            self.PZN = True

        # Else if clear
        elif commandLine[1] == "CLR":
            self.CLR = True

        # Else if preview
        elif commandLine[1] == "PRV":
            self.PRV = True

        # Else get aircraft ID
        else:

            # Get aircraft ID
            self.ACID = commandLine[1]

            # Get flight level
            if any(EFL.startswith("EFL") for EFL in commandLine):
                self.EFL = int(next(EFL[3:] for EFL in commandLine if "EFL" in EFL))

            # Get heading
            if any(HDG.startswith("HDG") for HDG in commandLine):
                self.HDG = int(next(HDG[3:] for HDG in commandLine if "HDG" in HDG))

            # Get speed
            if any(SPD.startswith("SPD") for SPD in commandLine):
                self.SPD = int(next(SPD[3:] for SPD in commandLine if "SPD" in SPD))

            # Get direct
            if any(DCT.startswith("DCT") for DCT in commandLine):
                self.DCT = True

            # Get transfer of control (TOC)
            if any(TOC.startswith("TOC") for TOC in commandLine):
                self.TOC = True

        # Get execute
        if any(EXQ.startswith("EXQ") for EXQ in commandLine):
            self.EXQ = True


class Run:
    # Initialization function
    def __init__(self, record_element, file_name):
        # self.advisories = record_element.find("./advisories").text
        self.date_time = record_element.find("./date_time").text
        # self.interactive = strToBool(record_element.find("./interactive").text)
        # self.performance = Performance(record_element.find("./performance"))
        self.scenario = Scenario(record_element.find("./scenario"))
        self.participant = record_element.find("./subject").text
        self.file_name = file_name
        self.workload = Workload(record_element.find("./workload"))
        self.commands = []
        self.logpoints = []
        for logpointElement in record_element.findall("./logpoint"):
            self.logpoints.append(Logpoint(logpointElement))

        #self.workloads = list()
        #for workloadElement in record_element.findall("./workload"):
        #    self.woworkloads.append(Workload(workloadElement))


class Workload():
    # Initialization function
    def __init__(self, workloadElement):
        self.ISA = list()
        for ISAElement in workloadElement.findall("./ISA"):
            self.ISA.append(ISA(ISAElement))


class ISA():
    # Initialization function
    def __init__(self, ISAElement):
        self.rating = float(ISAElement.find("./rating").text)
        self.response_time_s = float(ISAElement.find("./response_time_s").text)
        self.time_s = float(ISAElement.find("./time_s").text)


class Scenario():

    # Initialization function
    def __init__(self, scenarioElement):
        self.file = scenarioElement.find("./file").text
        self.airspace = Airspace(scenarioElement.find("./airspace"))
        self.settings = Settings(scenarioElement.find("./settings"))
        self.traffic = Traffic(scenarioElement.find("./traffic"), False)


class Settings():
    # Initialization function
    def __init__(self, settingsElement):
        self.icons = strToBool(settingsElement.find("./icons").text)
        self.LOA = float(settingsElement.find("./LOA").text)
        self.means_ends = strToBool(settingsElement.find("./means_ends").text)
        self.PZ = strToBool(settingsElement.find("./PZ").text)
        self.rotation = float(settingsElement.find("./rotation").text)
        self.simspeed = float(settingsElement.find("./simspeed").text)
        self.SSD = float(settingsElement.find("./SSD").text)


class Airspace():
    # Initialization function
    def __init__(self, airspaceElement):
        self.sectors = Sectors(airspaceElement.find("./sectors"))
        self.source_sinks = Sources_sinks(airspaceElement.find("./sources_sinks"))


class Sources_sinks():
    # Initialization function
    def __init__(self, sources_sinksElement):
        self.source_sink = list()
        for source_sinkElement in sources_sinksElement.findall("./source_sink"):
            self.source_sink.append(Source_sink(source_sinkElement))


class Source_sink():
    # Initialization function
    def __init__(self, source_sinkElement):
        self.point = Point(source_sinkElement.find("./point"))


class Sectors():
    # Initialization function
    def __init__(self, sectorsElement):
        self.sector = list()
        for sectorElement in sectorsElement.findall("./sector"):
            self.sector.append(Sector(sectorElement))


class Sector():
    # Initialization function
    def __init__(self, sectorElement):
        self.type = sectorElement.attrib["type"]
        self.border_points = Border_points(sectorElement.find("./border_points"))

class Border_points():
    # Initialization function
    def __init__(self, border_pointElement):
        self.point = list()
        for pointElement in border_pointElement.findall("./point"):
            self.point.append(Point(pointElement))


class Point():
    # Initialization function
    def __init__(self, pointElement):
        self.x_nm = float(pointElement.find("./x_nm").text)
        self.y_nm = float(pointElement.find("./y_nm").text)

        if pointElement.find("./name") is not None:
            self.name = pointElement.find("./name")
        else:
            self.name = None


class Logpoint():
    # Initialization function
    def __init__(self, logpointElement):
        self.score = float(logpointElement.find("./score").text)
        self.timestamp = logpointElement.find("./timestamp").text
        self.traffic = Traffic(logpointElement.find("./traffic"), True)


class Traffic():
    # Initialization function
    def __init__(self, trafficElement, isLogpoint):
        self.aircraft = list()
        for aircraftElement in trafficElement.findall("./aircraft"):
            if isLogpoint:
                self.aircraft.append(logpointAircraft(aircraftElement))
            else:
                self.aircraft.append(scenarioAircraft(aircraftElement))


class Performance():
    # Initialization function
    def __init__(self, performanceElement):
        self.average_score = float(performanceElement.find("./average_score").text)
        self.combined_cmds = float(performanceElement.find("./combined_cmds").text)
        self.midair_collision = strToBool(performanceElement.find("./midair_collision").text)
        self.speed_cmds = float(performanceElement.find("./speed_cmds").text)
        self.SSD_inspections = float(performanceElement.find("./SSD_inspections").text)
        self.track_cmds = float(performanceElement.find("./track_cmds").text)
        self.workload = float(performanceElement.find("./workload").text)


class Aircraft():
    # Initialization function
    def __init__(self, aircraftElement):
        self.ACID = aircraftElement.find("./ACID").text
        self.alt_cmd = float(aircraftElement.find("./alt_cmd").text)
        self.alt_ft = float(aircraftElement.find("./alt_ft").text)
        self.gam_deg = float(aircraftElement.find("./gam_deg").text)
        self.hdg_deg = float(aircraftElement.find("./hdg_deg").text)
        self.max_spd_kts = float(aircraftElement.find("./max_spd_kts").text)
        self.min_spd_kts = float(aircraftElement.find("./min_spd_kts").text)
        self.spd_kts = float(aircraftElement.find("./spd_kts").text)
        self.type = float(aircraftElement.find("./type").text)
        self.vs_fpm = float(aircraftElement.find("./vs_fpm").text)
        self.x_nm = float(aircraftElement.find("./x_nm").text)
        self.y_nm = float(aircraftElement.find("./y_nm").text)

    # Return aircraft position function
    def position(self):
        return [self.x_nm, self.y_nm]


class logpointAircraft(Aircraft):
    # Initialization function
    def __init__(self, aircraftElement):
        self.conflict = strToBool(aircraftElement.find("./conflict").text)
        self.controlled = strToBool(aircraftElement.find("./controlled").text)
        self.PZ_intrusion = strToBool(aircraftElement.find("./PZ_intrusion").text)
        self.PZ_intrusion_nm = float(aircraftElement.find("./PZ_intrusion_nm").text)
        self.selected = strToBool(aircraftElement.find("./selected").text)
        self.speed_cmd = float(aircraftElement.find("./speed_cmd").text)
        self.toc = strToBool(aircraftElement.find("./toc").text)
        self.track_cmd = float(aircraftElement.find("./track_cmd").text)
        super().__init__(aircraftElement)


class scenarioAircraft(Aircraft):
    # Initialization function
    def __init__(self, aircraftElement):
        self.COPX = aircraftElement.find("./COPX").text
        self.COPX_x_nm = float(aircraftElement.find("./COPX_x_nm").text)
        self.COPX_y_nm = float(aircraftElement.find("./COPX_y_nm").text)
        super().__init__(aircraftElement)



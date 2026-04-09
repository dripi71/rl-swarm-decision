from enum import Enum
# QObject: [agentId, eventType, actionTime]

class QObjectIndices(Enum):
    AGENTID = 0
    EVENTTYPE = 1
    ACTIONTIME = 2

class ActionTypes(Enum):
    NESTING = 0
    NESTING_FINISHED = 1
    SAMPLING = 2
    SAMPLING_FINISHED = 3
    LOCATION_EVENT = 4

class PredictionIndices(Enum):
    LOCATION = 0
    DURATION = 1
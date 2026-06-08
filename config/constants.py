from enum import IntEnum
# QObject: [agentId, eventType, actionTime]

class QObjectIndices(IntEnum):
    AGENTID = 0
    EVENTTYPE = 1
    ACTIONTIME = 2

class ActionTypes(IntEnum):
    NESTING = 0
    NESTING_FINISHED = 1
    SAMPLING = 2
    SAMPLING_FINISHED = 3
    PREDICT_ACTION = 4
    LOCATION_EVENT = 5

class PredictionKeys:
    LOCATION = "location"
    DURATION_PARAMS = "duration_params"
    VOTE = "vote"
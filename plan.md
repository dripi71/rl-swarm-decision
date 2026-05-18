**Observation Vector**

- für jede Location [timesteps,events]
- current confidence?
- current location
- wieviele Roboter sind mit mir im Nest?
- Die Bayesianischen Hyperparameter (a,b): Anstatt nur die Rohdaten zu senden, gib dem Agenten direkt die aktuellen Werte von a und b für jede Zone.
- Die "Stimmung" im Nest (Votes): Der Agent muss wissen, wie viele Stimmen aktuell für Blau und wie viele für Rot im Nest vorhanden sind. Ohne diese Information kann er keinen kollektiven Konsens lernen
- Eigener aktueller Vote: Welches Gebiet hält der Agent momentan selbst für das sicherere?
- action masking (brauche ich evtl doch nicht weil die agenten frei entscheiden sollen ob sie nach rot sampling direkt nochmal rot samplen wollen)

# Zu klärende Fragen:

# Eventuell besser das nest an stelle 0 ist? dann lernen die agenten besser dass 0 eine spezielle operation ist, sonst ist nest location immer ein verschiedner wert (index: num_loctions)

# - Was wenn 2 Aktionen zur gleichen Zeit? zb ein roboter leaved das nest und einer entered, wird dann zuerst geentered ( noch eine obs mehr im nest) oder geleaved?

# - wie genau soll die confidence berechnung durchgeführt werden bzw wann soll terminiert werden??
#
#
#
#
First Run: 2 Locations, 50 Agents, lambdas fixed auf [0.1, 0.8]!

!!! Bug in consensu berechnung! Aktuell nur abbruch wenn consensus = 100%
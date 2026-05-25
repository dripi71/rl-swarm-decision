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


Run 05: mit sehr wenig time decay (0.001)
Run 06: höherer time decay (0.05)




Problem:

Gerade wird bei den events aufgerundet wenn das nächste event innerhalb eines timesteps passiert.
Das hat aber das problem, dass ein Lambda von 10 und ein Lambda von 20 fast nicht zu unterscheiden sind weil beide jeden schritt ein event auslösen.

Lösung 1: 

Die schritte werden nicht aufgerundet sondern der timestep geschieht genau dann wen er passiert also auch in nicht ganzen schritten

Problem -> Laufzeit drastisch höher da viele events in der queue sind


Lösung 2:
Die Events sind nicht mehr in der Prio Q sondern jeder agent wenn er fertig ist mit samplen zieht aus dem entsprechenden
poisson prozess die anzahl events die er erfahren hat während seiner samplezeit

Problem: Nicht alle agents in einer Location sehen die gleichen events weil sie intern pro agent berechnet werden.

Lösung 3:

Man nimmt einfach kleinere Lambdas sodass die scale wieder passt



Vergleichsmetriken mit anderem Algorithmus:

steps
events experienced
lambdas
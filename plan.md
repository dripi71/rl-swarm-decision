**Observation Vector**

- für jede Location [timesteps,events]
- current confidence?
- current location
- wieviele Roboter sind mit mir im Nest?
- Die Bayesianischen Hyperparameter (a,b): Anstatt nur die Rohdaten zu senden, gib dem Agenten direkt die aktuellen Werte von a und b für jede Zone.
- Die "Stimmung" im Nest (Votes): Der Agent muss wissen, wie viele Stimmen aktuell für Blau und wie viele für Rot im Nest vorhanden sind. Ohne diese Information kann er keinen kollektiven Konsens lernen
- Eigener aktueller Vote: Welches Gebiet hält der Agent momentan selbst für das sicherere?
- action masking (brauche ich evtl doch nicht weil die agenten frei entscheiden sollen ob sie nach rot sampling direkt nochmal rot samplen wollen)



Vergleichsmetriken mit anderem Algorithmus:

steps_to_decision
correct_decision_rate

events_experienced_per_agent
total_events_until_decision
lambda_difficulty (zb.b lambdaSecond - lambda min) (geht nur für 2 locations)
quorum_achieved (yes / no)
lambdas


(!!!) eigentlich wird beim eval die logits greedy evaluiert! -> problem mit schwarm movement
Nächsten schritte: 

Runs evaluierbar machen:
- Mehrere Runs durchführbar machen
- Logs sammeln / Logs auswerten
- Plotten
- Mit random agenten vergleichen
- Mit Bayes Agenten vergleichen


Reward time decay: eventuell nicht linearer abfall, nach geeigneter Funktion suchen
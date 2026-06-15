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




Runs evaluierbar machen:
- Mehrere Runs durchführbar machen
- Logs sammeln / Logs auswerten
- Plotten
- Mit random agenten vergleichen
- Mit Bayes Agenten vergleichen


Reward time decay: eventuell nicht linearer abfall, nach geeigneter Funktion suchen

Parameter tuning pushing 80% acc:
- Wrong decision erhöhen auf -25 oder -30
- entropy coeff mit der zeit reduzieren
- learning time reduzieren
- 

Plan: Policies evaluieren -> rl parameter tuning
        - auf gleiche prior bringen!!
        - rl hauptimprovement: mehr sampling zulassen!
           -> weniger time decay, höhere max steps!
        Status 14:51 : 2 algos trainieren, einer gut einer instabil. Wenn fertig -> testen! 

      Währenddessen: dmmd + belief sharing implementieren
                     - experiment datei aufräumen
                     - runtime tests!
                     Status 14:51: runtime test läuft, muss noch ausgewertet dann (auswertdatei fertig), experimentdatei bisschen ausgeräumt.
                     

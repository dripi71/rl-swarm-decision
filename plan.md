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

!!! Achtung mit traveltime:

--- Step: 56 ---
Agents in Nest: 0
Agents in Sampling Locs: [0, 1]
Votes: [50, 0]
Lambdas: [0.1, 0.8]
Correct Agents: 0
Consensus: 0.0 [Needs: 0.7]

--- Step: 86 ---
Agents in Nest: 50
Agents in Sampling Locs: [0, 0]
Votes: [50, 0]
Lambdas: [0.1, 0.8]

Nach 30 schritten sind plötzlich alle im nest? Travel time eigentlich 10!



!! BUG:
Consensus wird erst berechnet wenn ein agent das nest verlässt (?)

--- Step: 56 ---
Agents in Nest: 0
Agents in Sampling Locs: [0, 2]
Votes: [50, 0]
Lambdas: [0.1, 0.8]
Correct Agents: 0
Consensus: 0.0 [Needs: 0.7]
Actions: {48: array([ 2, 19])}

--- Step: 56 ---
Agents in Nest: 0
Agents in Sampling Locs: [0, 1]
Votes: [50, 0]
Lambdas: [0.1, 0.8]
Correct Agents: 0
Consensus: 0.0 [Needs: 0.7]
Actions: {49: array([ 2, 19])}

--- Step: 86 ---
Agents in Nest: 50
Agents in Sampling Locs: [0, 0]
Votes: [50, 0]
Lambdas: [0.1, 0.8]
Correct Agents: 50
Consensus: 1.0 [Needs: 0.7]

-> Step 86 goal erreicht, aber erst nachdem alle agenten 56 (current step) + 10 (traveltime) + 19 (time at nest)
     -> sollte aber direkt bei eintritt ins nest eigentlich auch berechnet werden



! Wie kann man das beheben, wenn man es trotzdem nicht will? Wenn du möchtest, dass Agenten selbst bei gleichem Wissen von Natur aus völlig verschiedene Tendenzen haben (echter, dauerhafter Symmetriebruch), musst du den Agenten "Individualität" verleihen. Das macht man in MARL meist so, dass man die Beobachtung (Observation) anpasst:

Agent-ID übergeben: Du hängst an die Observation einfach den normierten Wert agent.id / num_agents an. Dadurch hat Agent 1 einen leicht anderen Input in das Netz als Agent 2, und das Netz wird für beide minimal unterschiedliche Wahrscheinlichkeiten ausspucken.
Zufallsrauschen (Noise): Man packt an das Ende jeder Observation einen kleinen Zufallswert.




Letztes Problem:

agenten können mehr reward  bekommen wenn sie trödeln -> sollten belohnt werden wenn sie schnell sind
# Q:

-> am beste doch kein heap weil nicht stabil und wenn man in eine sortierte liste einfügen möchte es bessere alternativen gibt
-> evtl binary search und python list
Payload dann sowas : [AgentId, Aktion(Id), zeit (zu filternder key)]

[type AgentId, type Action, type Start]

**Aktionen**:

- [AgentId, LEAVE_NEST, xxxxSteps]: Entferne Agent von der Pinlist, berechne zu welcher Location gegangen werden soll (RL) berechne ttSteps ->[AgentId, SAMPLING, xxxxSteps + ttSteps].
- [AgentId, SAMPLING, xxxxSteps]: berechne Sampling Länge slSteps (RL), berechne events (muss deterministisch sein! bzw die anderen roboter die samplen müssen auch die gleichen events erfahren), pushe auf Q [AgentId, GOTO_NEST, xxxxSteps + slSteps]
- [AgentId, GOTO_NEST, xxxxSteps]: berechne traveldauer tdSteps, füge [AgentId, NESTING, xxxxSteps + tdSteps]
- [AgentId, NESTING, xxxxSteps]: Füge Agent zur pinnwand hinzu, berechne Nesting time ntSteps, adde zur Q: [AgentId, LEAVE_NESt xxxxSteps + ntSteps]

**Observation Vector**

- für jede Location [timesteps,events]
- current confidence?
- current location
- wieviele Roboter sind mit mir im Nest?
- Die Bayesianischen Hyperparameter (a,b): Anstatt nur die Rohdaten zu senden, gib dem Agenten direkt die aktuellen Werte von a und b für jede Zone.
- Die "Stimmung" im Nest (Votes): Der Agent muss wissen, wie viele Stimmen aktuell für Blau und wie viele für Rot im Nest vorhanden sind. Ohne diese Information kann er keinen kollektiven Konsens lernen
- Eigener aktueller Vote: Welches Gebiet hält der Agent momentan selbst für das sicherere?

- **Action Space**

- Theoretisch muss der Roboter gar nicht strict location -> nest -> location -> nest befolgen
- Kann actions aussuchen nach belieben
- Action space wäre dann: [[Location1, Location2, ..., LocationN, NEST], maxWait]

# Zu klärende Fragen:

# - Was wenn 2 Aktionen zur gleichen Zeit? zb ein roboter leaved das nest und einer entered, wird dann zuerst geentered ( noch eine obs mehr im nest) oder geleaved?

# - wie genau soll die confidence berechnung durchgeführt werden

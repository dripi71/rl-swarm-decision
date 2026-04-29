TODO:
Add reward system

# Q:

[type AgentId, type ActionDone, type StepsDone]

**Aktionen**:

- [AgentId, LEAVE_NEST, xxxxSteps]: Entferne Agent von der Pinlist, berechne zu welcher Location gegangen werden soll (RL) berechne ttSteps ->[AgentId, SAMPLING, xxxxSteps + ttSteps].
- [AgentId, SAMPLING, xxxxSteps]: berechne Sampling Länge slSteps (RL), berechne events (muss deterministisch sein! bzw die anderen roboter die samplen müssen auch die gleichen events erfahren), pushe auf Q [AgentId, GOTO_NEST, xxxxSteps + slSteps]
  -> evtl Lösung: eine location event queue. Wenn zeit ausgelöst, dann hitte alle agents die in der location sind. Oder direkt in die haupt prio q einbauen
- [AgentId, GOTO_NEST, xxxxSteps]: berechne traveldauer tdSteps, füge [AgentId, NESTING, xxxxSteps + tdSteps]
- [AgentId, NESTING, xxxxSteps]: Füge Agent zur pinnwand hinzu, berechne Nesting time ntSteps, adde zur Q: [AgentId, LEAVE_NESt xxxxSteps + ntSteps]

- [AgentId, READYFORPREDICTION, xxxSteps]
  -> States unterscheiden in internal events und decision events

**Observation Vector**

- für jede Location [timesteps,events]
- current confidence?
- current location
- wieviele Roboter sind mit mir im Nest?
- Die Bayesianischen Hyperparameter (a,b): Anstatt nur die Rohdaten zu senden, gib dem Agenten direkt die aktuellen Werte von a und b für jede Zone.
- Die "Stimmung" im Nest (Votes): Der Agent muss wissen, wie viele Stimmen aktuell für Blau und wie viele für Rot im Nest vorhanden sind. Ohne diese Information kann er keinen kollektiven Konsens lernen
- Eigener aktueller Vote: Welches Gebiet hält der Agent momentan selbst für das sicherere?
- action masking (brauche ich evtl doch nicht weil die agenten frei entscheiden sollen ob sie nach rot sampling direkt nochmal rot samplen wollen)

- **Action Space**

- Theoretisch muss der Roboter gar nicht strict location -> nest -> location -> nest befolgen
- Kann actions aussuchen nach belieben
- Action space wäre dann: [[Location1, Location2, ..., LocationN, NEST], maxWait]

# Zu klärende Fragen:

# Eventuell besser das nest an stelle 0 ist? dann lernen die agenten besser dass 0 eine spezielle operation ist, sonst ist nest location immer ein verschiedner wert (index: num_loctions)

# - Was wenn 2 Aktionen zur gleichen Zeit? zb ein roboter leaved das nest und einer entered, wird dann zuerst geentered ( noch eine obs mehr im nest) oder geleaved?

# - wie genau soll die confidence berechnung durchgeführt werden bzw wann soll terminiert werden??

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
        - rl hauptimprovement: mehr sampling zulassen!
           -> weniger time decay, höhere max steps!
        Maßnahme jetzt: mehr timesteps, weniger time decay, höhere bestrafung bei falsch liegen
        Zukunft maßnahmen: voten bestrafen wenn er sich zu unsicher ist, auch wenn er richtig lägen würde


# In generateLambdas / reset: speichere die Permutation
self._loc_perm = np.random.permutation(num_locations)  # z.B. [1, 0] oder [0, 1]
self.lambdas = self.lambdas[self._loc_perm]

# In observe(): quality_score und andere loc-features entsprechend permutieren
quality_score = quality_score[self._loc_perm]
relative_uncertainty = relative_uncertainty[self._loc_perm]

# In step(): vote_action zurück-permutieren
actual_vote = self._loc_perm[vote_action]  # mappt Netz-Output auf echten Location-Index




TODOS vor Abgabe:
. firejob script entfernen aus repo (da ist meine uni email drin)
eval / policies ordner aus github repo hard löschen
- !Requirements aktualisieren, optuna ist neu dazu (tuning!)





































Problem: Seeding is off


Ursache: _sample_gamma_duration + stochastisches Action-Sampling verbrauchen unterschiedlich viele RNG-Aufrufe
Dein Seeding sieht auf den ersten Blick korrekt aus – set_global_seeds(episode_seed) setzt random, np.random und torch vor jeder Episode. Aber es gibt ein fundamentales Problem:

1. PyTorch-Sampling auf GPU vs CPU (Hauptursache)
In test_policy_against_baseline (Zeile 140–146) werden Actions per stochastischem Sampling erzeugt:

python
loc_dist = torch.distributions.Categorical(logits=loc_logits)
loc_actions = loc_dist.sample()    # ← nutzt torch RNG
vote_dist = torch.distributions.Categorical(logits=vote_logits)
vote_actions = vote_dist.sample()  # ← nutzt torch RNG
torch.manual_seed() setzt den CPU-RNG von PyTorch. Aber:

Auf dem Cluster (wahrscheinlich GPU vorhanden): Wenn PyTorch Tensoren auf CUDA liegen, nutzt sample() den GPU-RNG, der über torch.cuda.manual_seed() gesetzt wird. Der GPU-RNG erzeugt andere Sequenzen als der CPU-RNG, selbst bei gleichem Seed.
Auf deinem lokalen PC (wahrscheinlich nur CPU): sample() nutzt den CPU-RNG.
→ Unterschiedliche Actions → unterschiedliche _sample_gamma_duration-Aufrufe → unterschiedliche Events.

2. Unterschiedliche PyTorch-Versionen
Selbst wenn beide auf CPU laufen: Unterschiedliche PyTorch-Versionen (oder sogar Minor-Versionen) können unterschiedliche RNG-Implementierungen haben. Die interne Sequenz von torch.distributions.Categorical.sample() ist nicht über Versionen hinweg garantiert deterministisch.

3. Fehlende torch.use_deterministic_algorithms(True)
Du setzt nirgends:

python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
Ohne diese Flags kann PyTorch auch bei gleichem Seed nicht-deterministische Algorithmen wählen, die plattformabhängig unterschiedliche Ergebnisse liefern.

4. NumPy-RNG-Version
np.random.gamma() und np.random.exponential() werden in deiner Umgebung (_sample_gamma_duration, createLocationsAndAgents, Location-Events) häufig aufgerufen. Auch hier: Unterschiedliche NumPy-Versionen können bei gleichem Seed minimal andere Sequenzen erzeugen (NumPy hat den RNG in v1.x vs v2.x geändert).

Warum ist die Success-Rate trotzdem identisch?
Die Success-Rate hängt primär davon ab, ob die Policy "richtig" votet – und da die Policy deterministisch die gleiche Richtung angibt (die Logits sind identisch, die Argmax-Entscheidung ist stabil), kommt oft dieselbe finale Entscheidung raus. Die Events hingegen hängen von der exakten RNG-Sequenz ab (np.random.gamma, np.random.exponential), die nach einem einzigen abweichenden torch.sample()-Aufruf komplett auseinanderlaufen.

Fix-Empfehlung
Für plattformübergreifende Reproduzierbarkeit:

Forciere CPU-Inferenz explizit:

python
module = module.cpu()
obs_tensor = obs_tensor.cpu()
Nutze deterministische Algorithmen:

python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
Gleiche Paketversionen sicherstellen (PyTorch + NumPy) auf beiden Maschinen.

Alternativ: Nutze np.random.Generator statt des globalen np.random, um die NumPy-RNG-Sequenz von der PyTorch-Sequenz zu entkoppeln:

python
self.rng = np.random.default_rng(seed)
# dann: self.rng.exponential(...) statt np.random.exponential(...)
So beeinflussen PyTorch-RNG-Aufrufe nicht die NumPy-Event-Sequenz.

Soll ich einen dieser Fixes implementieren?


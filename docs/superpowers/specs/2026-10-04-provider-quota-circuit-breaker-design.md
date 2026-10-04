# Provider-Quota-Circuit-Breaker

**Status:** Entwurf vom Nutzer freigegeben und selbst geprüft
**Geltungsbereich:** Datenbankbasierte Tenant-Provider-Kaskade (Azure, OpenAI, OpenRouter und spätere Provider)

## Ziel

Dauerhafte Budget-/Quota-Sperren dürfen nicht bei jeder eingehenden Proxy-Anfrage erneut den gesperrten Upstream aufrufen. Der Proxy erkennt solche Sperren anhand strukturierter Providerfehler, persistiert den Breaker workerübergreifend und lässt nach einer Wartefrist höchstens eine echte Nutzeranfrage als Half-open-Probe zu. Transiente Rate-Limits und Kapazitätsprobleme bleiben von dauerhafter Quota-Erschöpfung getrennt.

Keine periodischen synthetischen Probeaufrufe: Bei ausbleibendem Nutztraffic wird nicht geprüft. Die Kaskade verwendet die nächste echte Anfrage für den Probeversuch.

## Architektur und Datenfluss

### Einheitliche Fehlerklassifikation

Ein providerneutraler Klassifikationsvertrag liefert eine Kategorie und optional einen scopesicheren Geltungsbereich sowie eine Retry-After-Dauer. Kategorien sind mindestens:

- `quota_exhausted`: eindeutiges, strukturiertes und dauerhaftes Budget-/Quota-Signal; öffnet den langlebigen Breaker.
- `transient`: vorübergehendes Rate-Limit, In-flight-Budget oder Kapazitätsfehler; öffnet keinen langlebigen Quota-Breaker und folgt der bestehenden kurzen Retry-/Failover-Policy.
- `terminal` und `unknown`: bestehende Fehler- und Failover-Regeln bleiben erhalten. Unbekannte oder mehrdeutige Fehler werden niemals als Quota-Erschöpfung geraten.

Provideradapter ordnen ausschließlich dokumentierte, strukturierte Fehlercodes/-felder zu. Freitext/Fehlermeldungen dienen nicht zur Klassifikation. HTTP-Status allein reicht nicht aus. Dieselbe Klassifikation gilt für HTTP-Fehler und strukturierte SSE-Fehler. Fehlerdetails und Credentials werden weder persistiert noch in den Clientfehler gespiegelt.

### Providerregeln

- **OpenAI:** Exakte Budgetcodes wie `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded` und `organization_usage_limit_exceeded` gelten als `quota_exhausted`. Das historische strukturierte `insufficient_quota` gilt ebenfalls als Quota-Signal. Normales Request-/Token-Rate-Limit ist transient.
- **OpenRouter:** Nur ein expliziter Key-Limit-Marker wie `error.metadata.limit_source=openrouter_key_limit` gilt als `quota_exhausted`. `openrouter_in_flight_budget` ist transient und verwendet einen gültigen `Retry-After`-Hinweis. Ein allgemeiner Credit-/402-Fehler ohne eindeutigen Key-Limit-Marker bleibt mehrdeutig und öffnet keinen Breaker.
- **Azure:** Allgemeine 429-, `rate_limit_exceeded`-, TPM/RPM- und Kapazitätsfehler sind transient; sie öffnen keine langlebige Quota-Sperre. Nur ein künftig belegbares, explizites Budgetsignal darf als `quota_exhausted` ergänzt werden.
- **Zukünftige Provider:** Sie integrieren sich über denselben Klassifikationsvertrag. Ohne explizite dokumentierte Quota-Regel wird ein Fehler nicht als langlebige Quota-Sperre behandelt.

### Breaker-Scope

Standard ist die einzelne Provider-Profil-ID. Eine profilübergreifende Sperre wird nur verwendet, wenn der Fehler selbst einen gemeinsamen Scope benennt und dieser durch eine explizit konfigurierte, passende Scope-ID eindeutig belegt ist. Beispiel: OpenAI-Organisationslimit auf exakt dieselbe konfigurierte Organisations-ID; Projektlimit auf exakt dieselbe Projekt-ID. Fehlt die ID oder kann die Zuordnung nicht sicher bestätigt werden, bleibt die Sperre profilbezogen. Secrets werden weder verglichen noch als Scope-Schlüssel verwendet.

Der persistierte Scope-Schlüssel ist ein stabiler, nicht umkehrbarer Fingerprint aus Tenant, Provider, Scope-Typ und expliziter Scope-ID. Der Fingerprint wird mit einem aus dem bestehenden Provider-Verschlüsselungsschlüssel getrennt abgeleiteten HMAC-Schlüssel gebildet. Providerfehlertexte, API-Schlüssel und Roh-Credentials werden nicht in der Breaker-Tabelle abgelegt.

### Persistenter Zustand und atomare Probe-Lease

PostgreSQL ist die Single Source of Truth für offene Breaker. Der Zustand umfasst mindestens Tenant/Provider/Scope-Fingerprint, stabile Fehlerkategorie, Fehler-/Backoff-Stufe, Öffnungs- bzw. nächste Probezeit, optionales Lease-Ablaufdatum sowie Aktualisierungszeit. Constraints und Indizes schützen die eindeutige Zuordnung und begrenzte Abfragen.

Beim Routing werden Profile mit offenem Breaker übersprungen; danach werden verfügbare Kandidaten in ihrer vorhandenen Kaskadenreihenfolge probiert. Ist die nächste Probe fällig, darf eine kurze atomare DB-Operation genau einer Anfrage die Half-open-Lease erteilen. Parallelrequests überspringen den Kandidaten. Eine Lease läuft nach einer festgelegten, begrenzten Frist aus, damit ein Worker-Absturz den Breaker nicht dauerhaft blockiert. Die Upstream-Anfrage läuft stets außerhalb einer Datenbanktransaktion.

Eine vom Breaker übersprungene Anfrage wird nicht als Provider-Versuch persistiert, weil kein Upstreamaufruf stattfand. Tatsächlich gestartete Probeaufrufe verwenden weiterhin das vorhandene Attempt-Lifecycle-Logging.

### Öffnen, Erholung und Backoff

Ein eindeutiger `quota_exhausted`-Fehler öffnet den Breaker sofort (keine Mehrfachfehler-Hysterese): weitere Requests während der Sperre könnten das Budget oder Rate-Limit weiter belasten. Erste Wiederprüfung frühestens nach einer Stunde. Scheitert die Half-open-Probe erneut eindeutig mit Quota-Erschöpfung, folgen Abstände von 2, 4, 8 und 16 Stunden, danach maximal 24 Stunden. Die Fehlerstufe wird je Scope persistiert. Ein valider providerseitiger `Retry-After`-Hinweis verschiebt den frühestmöglichen Probezeitpunkt weiter nach hinten; er verkürzt niemals den Backoff. Der Parser akzeptiert nur valide HTTP-Dauer-/Datumswerte und begrenzt extreme Werte auf eine definierte Sicherheitsobergrenze.

Die echte Half-open-Nutzeranfrage schließt den Breaker bei bestandenem begrenztem Preflight (Provider hat HTTP/SSE-Fehlerprüfung bestanden und die Antwort kann dem Client sicher übergeben werden). Ein später im Stream erkanntes eindeutiges Quota-Signal öffnet ihn erneut für folgende Requests; bereits gesendete Ausgabe wird nie wiederholt. Bei einer Probe mit anderem, nicht als Quota klassifiziertem Fehler wird die Quota-Sperre aufgehoben und die vorhandene Fehler-/Failover-Policy angewendet. Erfolgreicher Abschluss setzt Backoff-Stufe und Fehlerkategorie für diesen Scope zurück.

### Kaskade und keine verfügbaren Provider

Ein geblockter Provider wird vor jedem Upstreamaufruf übersprungen. Nicht gesperrte Kandidaten und bestehendes Failover-Verhalten bleiben unverändert. Falls alle Kandidaten gesperrt oder gerade exklusiv probe-belegt sind, antwortet der Proxy ohne Upstreamaufruf mit einem secretfreien `503` und `Retry-After` bis zum frühesten sinnvollen erneuten Prüfzeitpunkt. Gibt es parallel noch einen freien Kandidaten, wird dieser normal versucht; kein gesperrter Provider wird dabei vor seiner Probezeit aufgerufen.

Gewöhnliche transiente 429-/503-Fehler bleiben konzeptionell getrennt von der langlebigen Quota-Sperre und folgen der bestehenden kurzen Retry-/Failover-Policy. Bekannte transiente Fälle, darunter OpenRouter-In-Flight-Budget mit 402 plus Retry-After, müssen die providerseitige Wartezeit respektieren, ohne als dauerhafte Quota-Sperre behandelt zu werden.

### Dashboard

Die profilbezogene Anbieterstatusanzeige ergänzt einen klaren Zustand „Quota-Limit – pausiert bis …“ samt nächstem Probezeitpunkt. Shared-Scope-Sperren werden für alle betroffenen Profile konsistent sichtbar. UI/ARIA-Text bleibt verständlich und enthält weder Rohfehlertexte noch Credentials. Die vorhandene 5-Sekunden-HTMX-Aktualisierung wird wiederverwendet; Bewegungseinstellungen respektieren weiterhin `prefers-reduced-motion`.

## Migration und Kompatibilität

Eine neue Alembic-Migration legt persistente Breaker-Zustände mit PostgreSQL-spezifischen Constraints/Indizes an. Bestehende Profile und Requestverträge bleiben erhalten. Die Anwendung funktioniert weiter ohne DB-Breaker im bisherigen Single-/Environment-Auth-Modus; die Persistenz und Skip-Logik betreffen die datenbankbasierte Provider-Kaskade. Vorhandene Attempt-Ereignisse bleiben Betriebsereignisse und ersetzen den Breakerzustand nicht. Attempt-Telemetrie wird nach 24 Stunden opportunistisch beim Start eines neuen Versuchs in indexierten Batches bereinigt; diese Daten sind kein Langzeit-Audit.

## Fehler- und Transaktionsregeln

- Datenbank-Sperr-/Lease-Fehler dürfen keine unkontrollierte Upstream-Doppelprobe verursachen. Bei unklarem Lease-Ergebnis wird der betreffende Scope in diesem Request nicht aufgerufen; andere Kandidaten können weiter versucht werden.
- Zustandsänderungen sind kurze, atomare DB-Operationen. Keine Provider-HTTP-Aufrufe innerhalb einer DB-Transaktion.
- Lease-Freigabe/Abschluss ist idempotent und an den Scope sowie die konkrete Lease gebunden, damit ein verspäteter Worker keine neuere Probe überschreibt.
- Antwort-, Logging- und Persistenzpfade halten Providerfehler secretfrei. Kein Prompt-/Antwortinhalt kommt in Breaker-Datensätze oder Logs.
- Ein Breaker-Store-Ausfall wird nicht als „geschlossen“ fehlinterpretiert und führt nicht zu stillen Upstream-Aufrufen. Ein explizit konfiguriertes Profil ohne nachweisbare Breaker-Verfügbarkeit wird übersprungen; verfügbare andere Profile werden normal versucht.

## Abnahmekriterien

1. Exakte strukturierte OpenAI-Budgetcodes öffnen einen Breaker; gewöhnliches OpenAI-Rate-Limiting und unbekannte Codes tun dies nicht.
2. OpenRouter-Key-Limit öffnet; In-flight-Budget mit Retry-After bleibt transient; allgemeines/mehrdeutiges Credit-Signal öffnet nicht.
3. Azure-429-/`rate_limit_exceeded`-Signale öffnen keine langlebige Sperre.
4. Offene Profile erzeugen über mehrere Requests keine Upstreamaufrufe, bis Probezeit erreicht ist; Kaskaden-Fallback auf andere Profile funktioniert weiter.
5. Bei parallelen Prozessen erhält höchstens eine echte Anfrage pro fälligem Scope die Half-open-Lease; abgelaufene Leases sind wieder beanspruchbar.
6. Probe-Erfolg nach Preflight schließt und setzt den Backoff zurück; erneute Quota-Erschöpfung staffelt 1/2/4/8/16/24 Stunden und respektiert spätere valide Retry-After-Hinweise.
7. Späte SSE-Quota-Fehler beeinflussen künftige Requests, wiederholen jedoch keine bereits sichtbare Stream-Ausgabe.
8. Wenn alle Kandidaten offen/probe-belegt sind, wird ein sicherer `503` mit passendem `Retry-After` erzeugt und kein Upstream kontaktiert.
9. Profilbezogene und explizit gemeinsame Scope-Sperren werden korrekt isoliert; keine Sperre wird aus Credential-Gleichheit oder Freitext abgeleitet.
10. Dashboardstatus, nächster Probezeitpunkt, Barrierefreiheit und HTMX-Refresh sind getestet.
11. Migration, PostgreSQL-Constraints, Transaktions-/Konkurrenzverhalten, Single-/Environment-Modus und bestehendes Failover-/Attempt-Lifecycle-Verhalten sind abgedeckt.
12. Kein Secret, Rohfehlertext, Prompt oder Antwortinhalt wird im Breakerzustand gespeichert oder dargestellt.

## Referenzen

- [Azure OpenAI Quota und 429-Drosselung](https://learn.microsoft.com/azure/foundry/openai/how-to/quota#understanding-429-throttling-errors-and-what-to-do)
- [OpenAI Spend Limits](https://developers.openai.com/api/docs/guides/spend-limits)
- [OpenAI Rate Limits](https://developers.openai.com/api/docs/guides/rate-limits)
- [OpenRouter Credit und Rate Limits](https://openrouter.ai/docs/api_reference/limits)
- [OpenRouter Error Handling](https://openrouter.ai/docs/api_reference/errors-and-debugging)

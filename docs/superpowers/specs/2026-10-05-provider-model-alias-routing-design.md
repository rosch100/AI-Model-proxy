# Provider-Modellrouting nach Modell-ID und Kontoalias

Datum: 2026-10-05
Status: zur Implementierung freigegeben
Scope: datenbankbasierte Tenant-Providerroute (`/v1` und `/azure/v1`)

## Zielverhalten

- Sendet Cursor die Tenant-Cursor-ID, bleibt das bisherige Verhalten erhalten: Profile werden nach Routenpriorität mit ihren Standardmodellen versucht.
- Sendet Cursor stattdessen eine native Modell-ID aus einem aktiven Profilkatalog, werden ausschließlich Profile versucht, deren Katalog genau diese ID enthält. Bei Duplikaten entscheidet die bestehende Routenpriorität; Failover bleibt auf diese passenden Konten beschränkt.
- Eine kontospezifische ID hat die Form `Kontoname/Modell-ID`, beispielsweise `Research Team/deepseek-flash`. Bei eindeutigem Kontonamen ist das Konto gepinnt. Falls Kontonamen nicht eindeutig sind, steht zusätzlich der Provider davor: `provider:Kontoname/Modell-ID` (etwa `openrouter:Research Team/anthropic/model-x`).
- Konto-Whitespace bleibt unverändert und lesbar. Im Kontonamen werden reservierte Trennzeichen und Nicht-ASCII-Zeichen percent-encoded (`urllib.parse.quote(..., safe=" ")`); native Modell-IDs bleiben unverändert. Die Auflösung erfolgt durch Vergleich gegen aus Profilmetadaten und Katalogen erzeugte IDs, nicht durch mehrdeutiges Splitten des Modellstrings. OpenRouter-Modellpfade mit `/` bleiben daher gültig.
- Ein eindeutiger Kontoname oder ein providerqualifizierter Alias identifiziert genau ein Profil/Modell und bindet die Anfrage an dieses Profil. Upstream-Fehler, Quota und Circuit-Breaker lösen keinen Wechsel zu einem anderen Profil aus.
- Die Tenant-Modellliste enthält die eindeutigen nativen IDs plus alle qualifizierten IDs; die bisherige Tenant-Cursor-ID bleibt enthalten. Azure-only-Modelllisten enthalten nur Azure-Profile.
- Unbekannte oder nicht aktive IDs werden lokal mit HTTP 400 abgewiesen; es gibt keinen impliziten Wechsel zu einem Standardmodell.

## Architektur

Die zentrale Auflösung liegt in `app/providers/routing.py`. `app/providers/model_ids.py` erzeugt Kontoaliases und providerqualifizierte IDs; dieselben Funktionen werden von Routing und `/v1/models`-Projektion verwendet. Ein nativer Modellname filtert die Kandidatenliste auf Katalogtreffer. Ein Kontoalias wird nur bei eindeutigem Kontonamen ausgegeben und pinnt genau dieses Profil; ein providerqualifizierter Alias erlaubt auch bei Namensduplikaten eine eindeutige Auswahl. Nur die bestehende Tenant-Cursor-ID durchläuft alle Profile.

Die Kataloge und Profilnamen aus dem bereits konsistent geladenen `DatabaseTenantRoutingSnapshot` bleiben die einzige Quelle; es ist keine neue Konfiguration oder Datenbanktabelle nötig. Da qualifizierte IDs länger als native Modell-IDs sein können, werden die `inbound_model`-Spalten für Providerattempts und Inferenceaktivität auf 2048 Zeichen erweitert. Eine PostgreSQL-Migration sichert das Live-Schema ab.

## Fehler- und Kompatibilitätsverhalten

- Gleiche native Modell-ID auf mehreren Profilen: geordneter Failover nur zwischen diesen Treffern.
- Qualifizierter Treffer: einzelnes Profil; kein Profil-Failover.
- Kein Treffer: lokaler Cursor-Konfigurationsfehler vor Upstreamaufruf.
- Fehlender Profilname: native Modell-ID bleibt nutzbar; für dieses Profil wird kein Kontoalias ausgegeben.
- Die alte Tenant-Cursor-ID und Nicht-Tenant-/Codex-Modelllisten bleiben unverändert.
- Fehlende Namens-/Modellkonfiguration wird nicht durch einen Dummywert ersetzt.

## Tests und Abnahme

- Modell-ID-Projektion: native, doppelte und qualifizierte IDs; Leerzeichen, reservierte Zeichen, OpenRouter-Modellpfade; Azure-only-Filter; unveränderte Cursor-ID.
- Auflösung: native Einzel-/Mehrfachtreffer, exakter Kontoalias, nicht übereinstimmende Konten/Provider, unbekannte ID und Erhalt der Custom-ID-Kaskade.
- Routing: native Modell-ID kontaktiert keine Profile ohne Katalogtreffer; qualifizierter Fehler versucht kein zweites Profil; Circuit-/Attempt-Daten bleiben auf das gewählte Ziel bezogen.
- PostgreSQL-Migration ändert beide Inbound-ID-Spalten auf `VARCHAR(2048)`; DDL-/Schema-Tests bestätigen die Länge.
- Gezielte Routing-/Modelllisten-/Aktivitätstests sowie vollständige pytest- und Lint-Suite bestehen.

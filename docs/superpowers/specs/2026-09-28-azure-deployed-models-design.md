# Azure-Deployed-Modelle im Cursor-Proxy

## Ziel

Alle Responses-fähigen Modell-Deployments des bereits verwendeten Azure-Kontos `AzureOpenAI-Instanz2` sollen als auswählbare Cursor-Modelle durch den Proxy erreichbar sein.

## Azure-Inventar

Die Azure-Abfrage vom 28.09.2026 meldet sechs erfolgreiche Deployments mit `responses=true`:

| Cursor-Modell-ID | Azure-Deployment |
|---|---|
| `gpt-5.6-luna` | `gpt-5-6-luna-api` |
| `gpt-5.6-sol` | `gpt-5-6-sol-api` |
| `gpt-5.6-terra` | `gpt-5-6-terra-api` |
| `gpt-6-astra` | `gpt-6-astra-api` |
| `gpt-6-luna` | `gpt-6-luna-api` |
| `gpt-6-sol` | `gpt-6-sol-api` |

`gpt-5.6-sol` und `gpt-6-luna` sind bereits im Proxy-Katalog. Die anderen vier Modell-IDs fehlen.

## Entscheidung

Die bestehende explizite Allowlist bleibt die Quelle der unterstützten Cursor-Modell-IDs. Die vier fehlenden IDs werden ergänzt; `.env.example` bildet alle sechs IDs auf die tatsächlichen Deployment-Namen ab. Die bestehende Request-Adapter- und Responses-API-Architektur bleibt unverändert.

Alternativen:

- **Alle Azure-Deployments dynamisch freischalten:** verworfen, weil dadurch die Allowlist umgangen würde und die Proxy-Kompatibilität nicht für beliebige Deployment-Typen garantiert ist.
- **Deployment-Namen als Cursor-Modell-IDs verwenden:** verworfen, weil Deployment-Namen kontospezifisch sind und nicht den Modell-IDs entsprechen, die der Proxy validiert.
- **Deployment-Zuordnungen im Python-Code fest verdrahten:** verworfen, weil die Azure-Namen zur Konto-Konfiguration gehören und über `AZURE_MODEL_DEPLOYMENTS` konfigurierbar bleiben sollen.

## Umfang und Grenzen

- Der Scope umfasst die sechs Deployments in `AzureOpenAI-Instanz2`. Andere Azure-Konten benötigen eine eigene Base-URL und sind nicht Teil dieser Zuordnung.
- Kein neues Azure-SDK und keine Änderung am Transportprotokoll; der Proxy verwendet weiter Azure Responses API.
- Die Azure-Metadaten belegen erfolgreiche Deployments und `responses=true`. Eine Live-Inferenz gegen Azure ist kein Teil des lokalen Regressionstests.

## Akzeptanz

1. `/v1/models` listet alle sechs Modell-IDs.
2. Jeder Cursor-Modellname wird über `AZURE_MODEL_DEPLOYMENTS` an das korrekte Deployment geroutet.
3. Für alle sechs Modelle belegen Regressionstests das Routing; Reasoning-Suffixe bleiben über den bestehenden Adapter nutzbar.
4. `.env.example` und README dokumentieren die sechs echten Deployment-Zuordnungen und kennzeichnen fehlende Proxy-End-to-End-Verifikation korrekt.

## Schnittstelleninventar

- **`azure-model-catalog`** — `contract`; `SUPPORTED_MODELS` speist den authentifizierten `/v1/models`-Endpunkt; Nachweis: `tests/test_models.py::TestModels::test_models_endpoint_returns_200`.
- **`azure-deployment-routing`** — `contract`; `AZURE_MODEL_DEPLOYMENTS` ordnet die Cursor-Modell-ID dem Azure-Deployment zu; Nachweis: parametrisierte Routing-Tests in `tests/test_request_adapter.py`.

## Offene Lücke

- Live-Inferenz gegen Azure wird nicht ausgeführt. Der Azure-Inventaraufruf belegt `Succeeded` und `responses=true`, aber nicht den Ende-zu-Ende-Aufruf aus Cursor über den Proxy.

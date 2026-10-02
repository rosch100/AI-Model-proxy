# Operating the Cursor Azure proxy for a team

This proxy translates Cursor-compatible requests to Azure OpenAI's Responses API and streams adapted responses back to Cursor. A small Linux VM is a reasonable deployment option for a trusted team, provided an operator manages updates, availability, network security, and Azure capacity.

## Choose an authentication mode

- Use `AUTH_MODE=single` for one trusted group when sharing one Cursor API key and one Azure configuration is acceptable.
- Use `AUTH_MODE=tenant` when groups need separate API keys and Azure resources/deployment maps. Tenant mode isolates routing and conversation cache keys, but does not provide per-person accounts, budgets, billing, or usage attribution.
- Keep Codex disabled for tenant mode. The Codex provider uses shared local authentication and is not designed as a multi-tenant team backend.

## Deployment checklist

- Run the production Docker Compose service and keep its application port bound to loopback. Put a maintained reverse proxy such as Caddy or Nginx in front to provide public HTTPS.
- Restrict inbound firewall access to HTTPS and administrative SSH; use key-based SSH access and keep the host and containers patched.
- Supply API keys and tenant configuration at runtime through protected environment/secrets. Do not commit `.env` or bake credentials into images.
- Keep `LOG_COMPLETION=off` when prompts or responses may contain sensitive content. Review log and traffic-recording settings before deployment and define a retention policy.
- Configure worker count and VM sizing for measured concurrent load. More proxy workers do not increase Azure RPM/TPM quotas; coordinate capacity and cost limits with the Azure deployment owner.
- Monitor proxy health, Azure throttling, latency, and costs. Test a deployment with representative parallel Cursor requests before onboarding a team.

## Cursor configuration

In Cursor's OpenAI-compatible provider settings, set the Override OpenAI Base URL to the public HTTPS endpoint and use the configured single-mode key or the relevant tenant's cleartext API key. Use a model ID exposed by the proxy and mapped to an Azure deployment for the selected principal.

See the repository [README](../README.md) for authentication, model mapping, configuration, and smoke-test details. For a deployment-specific procedure, keep hostnames, resource names, and secret values in operator-managed configuration rather than in this general guide.

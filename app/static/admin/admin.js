document.documentElement.classList.add("js");

document.querySelectorAll("[data-reveal-secret]").forEach((checkbox) => {
  checkbox.addEventListener("change", () => {
    const input = document.getElementById(checkbox.dataset.revealSecret);
    if (!input) {
      return;
    }
    input.type = checkbox.checked ? "text" : "password";
  });
});

const localTimeFormatter = new Intl.DateTimeFormat(
  document.documentElement.lang || undefined,
  { dateStyle: "medium", timeStyle: "medium" },
);

function initializeLocalTimes() {
  document.querySelectorAll("time[data-local-time]").forEach((element) => {
    const timestamp = new Date(element.dateTime);
    if (!Number.isNaN(timestamp.getTime())) {
      element.textContent = localTimeFormatter.format(timestamp);
      element.title = Intl.DateTimeFormat().resolvedOptions().timeZone;
    }
  });
}

initializeLocalTimes();

function initializeAzureScopeLinks() {
  document.querySelectorAll("[data-azure-scope-form]").forEach((form) => {
    const subscription = form.querySelector('[name="subscription_id"]');
    const resourceGroup = form.querySelector('[name="resource_group_arm_id"]');
    const cognitiveResource = form.querySelector(
      '[name="cognitive_resource_arm_id"]',
    );
    const groupLink = form.querySelector(
      '[data-azure-scope-link="resource-group"]',
    );
    const resourceLink = form.querySelector(
      '[data-azure-scope-link="cognitive-resource"]',
    );
    const subscriptionPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
    const resourceNamePattern = /^[\p{Lu}\p{Ll}\p{Lt}\p{Lm}\p{Lo}\p{Nd}_.()()-]{1,90}$/u;
    const updateLinks = () => {
      const validSubscription = subscription.value.trim();
      const groupMatch = resourceGroup.value.trim().match(
        /^\/subscriptions\/([^/]+)\/resourceGroups\/([^/?#]+)$/i,
      );
      if (
        groupMatch &&
        subscriptionPattern.test(groupMatch[1]) &&
        resourceNamePattern.test(groupMatch[2]) &&
        !groupMatch[2].endsWith(".") &&
        groupMatch[1].toLowerCase() === validSubscription.toLowerCase() &&
        subscriptionPattern.test(validSubscription)
      ) {
        const resourceId = `/subscriptions/${groupMatch[1]}/resourceGroups/${groupMatch[2]}`;
        groupLink.href = `https://portal.azure.com/#/resource${resourceId}/overview`;
        groupLink.hidden = false;
      } else {
        groupLink.removeAttribute("href");
        groupLink.hidden = true;
      }

      const resourceMatch = cognitiveResource.value.trim().match(
        /^\/subscriptions\/([^/]+)\/resourceGroups\/([^/]+)\/providers\/Microsoft\.CognitiveServices\/accounts\/([^/?#]+)$/i,
      );
      if (
        resourceMatch &&
        groupMatch &&
        subscriptionPattern.test(resourceMatch[1]) &&
        resourceNamePattern.test(resourceMatch[2]) &&
        !resourceMatch[2].endsWith(".") &&
        /^[a-z0-9-]{2,64}$/i.test(resourceMatch[3]) &&
        resourceMatch[1].toLowerCase() === validSubscription.toLowerCase() &&
        resourceMatch[2].toLowerCase() === groupMatch[2].toLowerCase()
      ) {
        const resourceId = `/subscriptions/${resourceMatch[1]}/resourceGroups/${resourceMatch[2]}/providers/Microsoft.CognitiveServices/accounts/${resourceMatch[3]}`;
        resourceLink.href = `https://portal.azure.com/#/resource${resourceId}/overview`;
        resourceLink.hidden = false;
      } else {
        resourceLink.removeAttribute("href");
        resourceLink.hidden = true;
      }
    };
    [subscription, resourceGroup, cognitiveResource].forEach((input) => {
      input.addEventListener("input", updateLinks);
    });
    updateLinks();
  });
}

initializeAzureScopeLinks();

document.body.addEventListener("change", (event) => {
  if (event.target.matches("[data-activity-period]")) {
    event.target.form.requestSubmit();
  }
});

document.body.addEventListener("htmx:afterSwap", () => {
  initializeLocalTimes();
});

const providerSelect = document.querySelector("[data-provider-switch]");
if (providerSelect) {
  const providerGroups = document.querySelectorAll("[data-provider-fields]");
  const updateProviderFields = () => {
    providerGroups.forEach((group) => {
      group.hidden = group.dataset.providerFields !== providerSelect.value;
    });
  };
  providerSelect.addEventListener("change", updateProviderFields);
  updateProviderFields();
}

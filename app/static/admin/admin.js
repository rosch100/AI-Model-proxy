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

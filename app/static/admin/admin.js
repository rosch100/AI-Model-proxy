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

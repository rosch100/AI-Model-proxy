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

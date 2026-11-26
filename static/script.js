// Reserved for small frontend enhancements.
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("trafficForm");
  if (form) {
    form.addEventListener("submit", function () {
      const button = form.querySelector("button[type='submit']");
      button.innerHTML = "Analysing conditions <span>…</span>";
      button.disabled = true;
    });
  }
});

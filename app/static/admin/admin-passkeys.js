(() => {
  const csrfMeta = document.querySelector('meta[name="csrf-token"]');

  function csrfToken() {
    return csrfMeta ? csrfMeta.getAttribute("content") : "";
  }

  function bufferToBase64url(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    bytes.forEach((b) => {
      binary += String.fromCharCode(b);
    });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
  }

  function base64urlToBuffer(value) {
    const padded = value.replace(/-/g, "+").replace(/_/g, "/");
    const pad = padded.length % 4 === 0 ? "" : "=".repeat(4 - (padded.length % 4));
    const binary = atob(padded + pad);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  function publicKeyOptionsFromJson(options) {
    const copy = structuredClone(options);
    copy.challenge = base64urlToBuffer(options.challenge);
    if (copy.user && copy.user.id) {
      copy.user.id = base64urlToBuffer(copy.user.id);
    }
    if (Array.isArray(copy.excludeCredentials)) {
      copy.excludeCredentials = copy.excludeCredentials.map((item) => ({
        ...item,
        id: base64urlToBuffer(item.id),
      }));
    }
    if (Array.isArray(copy.allowCredentials)) {
      copy.allowCredentials = copy.allowCredentials.map((item) => ({
        ...item,
        id: base64urlToBuffer(item.id),
      }));
    }
    return copy;
  }

  function credentialToJson(credential) {
    const response = credential.response;
    const payload = {
      id: credential.id,
      rawId: bufferToBase64url(credential.rawId),
      type: credential.type,
      response: {
        clientDataJSON: bufferToBase64url(response.clientDataJSON),
      },
    };
    if (response.attestationObject) {
      payload.response.attestationObject = bufferToBase64url(response.attestationObject);
    }
    if (response.authenticatorData) {
      payload.response.authenticatorData = bufferToBase64url(response.authenticatorData);
    }
    if (response.signature) {
      payload.response.signature = bufferToBase64url(response.signature);
    }
    if (response.userHandle) {
      payload.response.userHandle = bufferToBase64url(response.userHandle);
    }
    if (credential.authenticatorAttachment) {
      payload.authenticatorAttachment = credential.authenticatorAttachment;
    }
    return payload;
  }

  async function postJson(url, body) {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrfToken(),
      },
      credentials: "same-origin",
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error || "Anfrage fehlgeschlagen.");
    }
    return data;
  }

  function setStatus(id, message, isError) {
    const node = document.getElementById(id);
    if (!node) {
      return;
    }
    node.hidden = !message;
    node.textContent = message || "";
    node.classList.toggle("error", Boolean(isError));
  }

  async function passkeyLogin({ conditional = false } = {}) {
    if (!conditional) {
      setStatus("passkey-login-status", "Passkey wird angefragt…", false);
    }
    const begin = await postJson("/admin/webauthn/login/begin", {});
    const requestOptions = {
      publicKey: publicKeyOptionsFromJson(begin.options),
    };
    if (conditional) {
      requestOptions.mediation = "conditional";
    }
    const credential = await navigator.credentials.get(requestOptions);
    if (!credential) {
      throw new Error("Kein Passkey ausgewählt.");
    }
    const complete = await postJson("/admin/webauthn/login/complete", {
      challenge_id: begin.challenge_id,
      credential: credentialToJson(credential),
    });
    window.location.href = complete.redirect || "/admin/";
  }

  async function passkeyRegister(label) {
    setStatus("passkey-register-status", "Registrierung startet…", false);
    const begin = await postJson("/admin/webauthn/register/begin", {});
    const credential = await navigator.credentials.create({
      publicKey: publicKeyOptionsFromJson(begin.options),
    });
    if (!credential) {
      throw new Error("Registrierung abgebrochen.");
    }
    const complete = await postJson("/admin/webauthn/register/complete", {
      challenge_id: begin.challenge_id,
      credential: credentialToJson(credential),
      label: label || "Passkey",
    });
    window.location.href = complete.redirect || "/admin/";
  }

  document.addEventListener("DOMContentLoaded", () => {
    const loginButton = document.getElementById("passkey-login");
    if (loginButton) {
      loginButton.addEventListener("click", () => {
        passkeyLogin().catch((error) => {
          setStatus("passkey-login-status", error.message || String(error), true);
        });
      });
      if (
        window.PublicKeyCredential &&
        typeof PublicKeyCredential.isConditionalMediationAvailable === "function"
      ) {
        PublicKeyCredential.isConditionalMediationAvailable().then((available) => {
          if (!available) {
            return;
          }
          passkeyLogin({ conditional: true }).catch(() => {
            // Conditional mediation is best-effort; the button remains available.
          });
        });
      }
    }
    const registerButton = document.getElementById("passkey-register");
    if (registerButton) {
      registerButton.addEventListener("click", () => {
        const labelInput = document.getElementById("passkey-label");
        const label = labelInput ? labelInput.value : "Passkey";
        passkeyRegister(label).catch((error) => {
          setStatus("passkey-register-status", error.message || String(error), true);
        });
      });
    }
  });
})();

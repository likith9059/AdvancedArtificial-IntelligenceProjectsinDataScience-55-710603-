(() => {
    const input = document.getElementById("image-input");
    const selectedFile = document.getElementById("selected-file");
    const dropZone = document.getElementById("drop-zone");
    const form = document.getElementById("prediction-form");
    const button = document.getElementById("inspect-button");

    if (!input || !dropZone || !form || !button) return;

    const showFilename = () => {
        const file = input.files && input.files[0];
        selectedFile.textContent = file ? file.name : "No file selected";
    };

    input.addEventListener("change", showFilename);

    ["dragenter", "dragover"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.add("dragging");
        });
    });

    ["dragleave", "drop"].forEach((eventName) => {
        dropZone.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropZone.classList.remove("dragging");
        });
    });

    dropZone.addEventListener("drop", (event) => {
        const files = event.dataTransfer && event.dataTransfer.files;
        if (!files || !files.length) return;
        const transfer = new DataTransfer();
        transfer.items.add(files[0]);
        input.files = transfer.files;
        showFilename();
    });

    form.addEventListener("submit", () => {
        if (input.files && input.files.length) {
            button.classList.add("loading");
            button.disabled = true;
        }
    });
})();

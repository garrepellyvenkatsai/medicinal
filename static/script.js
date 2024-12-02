

document.getElementById("uploadForm").addEventListener("submit", async function (event) {
    event.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById("imageUpload");
    formData.append("file", fileInput.files[0]);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData,
    });

    const data = await response.json();

    if (data.error) {
        alert(data.error);
    } else {
        document.getElementById("uploadedImage").src = data.image_url;
        document.getElementById("plantName").textContent = `Plant Name: ${data.plant_name}`;
        document.getElementById("plantUses").textContent = `Uses: ${data.details.uses}`;
        document.getElementById("plantBenefits").textContent = `Benefits: ${data.details.benefits}`;
        document.getElementById("plantLocations").textContent = `Locations: ${data.details.locations}`;
    }
});

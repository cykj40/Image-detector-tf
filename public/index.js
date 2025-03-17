import { resetCanvas } from "./utils.js";

const clearButton = document.getElementById("clear-button");

clearButton.addEventListener("click", () => {
    console.log("Clear button clicked");
    resetCanvas();

    const predictionParagraph = document.querySelector(".prediction");
    if (predictionParagraph) {
        predictionParagraph.textContent = "";
        console.log("Prediction text cleared");
    } else {
        console.log("Prediction paragraph not found");
    }
});
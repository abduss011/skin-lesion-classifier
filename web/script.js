const LABELS = [
  "Actinic keratoses (akiec)",
  "Basal cell carcinoma (bcc)",
  "Benign keratosis-like lesions (bkl)",
  "Dermatofibroma (df)",
  "Melanoma (mel)",
  "Melanocytic nevi (nv)",
  "Vascular lesions (vasc)",
];

const MODEL_PATH = "../web_model/model.json";
const IMAGE_SIZE = 224;

let model = null;
let imageLoaded = false;

const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const predictBtn = document.getElementById("predict-btn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const topClassEl = document.getElementById("top-class");
const probsBodyEl = document.getElementById("probs-body");

async function loadModel() {
  try {
    statusEl.textContent = "Loading model...";
    console.log("Loading model from:", MODEL_PATH);
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log("Model loaded successfully!");
    if (model.inputs && model.inputs.length > 0) {
      console.log("Model input shape:", model.inputs[0].shape);
    }
    if (model.outputs && model.outputs.length > 0) {
      console.log("Model output shape:", model.outputs[0].shape);
    }
    statusEl.textContent = "Model loaded. Choose an image.";
    predictBtn.disabled = !imageLoaded;
  } catch (err) {
    console.error("Model loading error:", err);
    statusEl.textContent = "Failed to load model. Check console for details.";
    alert("Failed to load model. Please check:\n1. Model files are in web_model/\n2. You're running a local server (not file://)\n3. Check browser console for details");
  }
}

function handleFileChange(event) {
  const file = event.target.files[0];
  if (!file) {
    imageLoaded = false;
    previewImg.style.display = "none";
    previewPlaceholder.style.display = "flex";
    predictBtn.disabled = true;
    resultsEl.classList.add("hidden");
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.onload = () => {
      imageLoaded = true;
      previewImg.style.display = "block";
      previewPlaceholder.style.display = "none";
      console.log("Image loaded. Dimensions:", previewImg.width, "x", previewImg.height);
      predictBtn.disabled = !model;
      resultsEl.classList.add("hidden");
    };
    previewImg.onerror = () => {
      console.error("Failed to load image");
      imageLoaded = false;
      alert("Failed to load image. Please try another file.");
    };
  };
  reader.readAsDataURL(file);
}

function preprocessImage(img) {
  return tf.tidy(() => {
    let tensor = tf.browser.fromPixels(img).toFloat();
    console.log("Original image shape:", tensor.shape);
    tensor = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);
    console.log("After resize:", tensor.shape);
    tensor = tensor.div(127.5).sub(1.0);
    const minVal = tensor.min().dataSync()[0];
    const maxVal = tensor.max().dataSync()[0];
    console.log("Preprocessed tensor range:", minVal, "to", maxVal);
    
    if (isNaN(minVal) || isNaN(maxVal)) {
      throw new Error("Preprocessing produced NaN values");
    }
    const batched = tensor.expandDims(0);
    console.log("Final input shape:", batched.shape);
    return batched;
  });
}

async function runPrediction() {
  if (!model || !imageLoaded) {
    console.error("Model or image not ready");
    return;
  }

  statusEl.textContent = "Running prediction...";
  predictBtn.disabled = true;

  try {
    const input = preprocessImage(previewImg);
    console.log("Input shape:", input.shape);
    
    const prediction = model.predict(input);
    console.log("Prediction shape:", prediction.shape);
    console.log("Prediction:", prediction);
    let probs;
    if (prediction instanceof tf.Tensor) {
      probs = await prediction.data();
      const shape = prediction.shape;
      console.log("Output shape:", shape);
      if (shape.length === 2 && shape[0] === 1) {
        probs = Array.from(probs);
      } else {
        probs = Array.from(probs);
      }
    } else {
      probs = Array.from(prediction);
    }
    
    console.log("Probabilities:", probs);
    console.log("Number of classes:", probs.length);
    
    if (probs.some(p => isNaN(p) || !isFinite(p))) {
      throw new Error("Model output contains NaN or invalid values");
    }
    if (probs.length !== LABELS.length) {
      console.warn(`Expected ${LABELS.length} classes, got ${probs.length}`);
      while (probs.length < LABELS.length) {
        probs.push(0);
      }
      probs = probs.slice(0, LABELS.length);
    }
    const pairs = probs.map((p, i) => ({ 
      label: LABELS[i] || `Class ${i}`, 
      prob: Number(p) 
    }));
    pairs.sort((a, b) => b.prob - a.prob);

    const top = pairs[0];
    if (isNaN(top.prob) || !isFinite(top.prob)) {
      throw new Error("Top prediction is invalid");
    }
    
    topClassEl.innerHTML = `<span class="label">${top.label}</span> &ndash; ${(top.prob * 100).toFixed(1)}%`;

    probsBodyEl.innerHTML = "";
    pairs.forEach((item) => {
      const tr = document.createElement("tr");
      const nameTd = document.createElement("td");
      const probTd = document.createElement("td");
      nameTd.textContent = item.label;
      const probValue = isNaN(item.prob) || !isFinite(item.prob) ? "0.0" : (item.prob * 100).toFixed(1);
      probTd.textContent = probValue + " %";
      tr.appendChild(nameTd);
      tr.appendChild(probTd);
      probsBodyEl.appendChild(tr);
    });

    resultsEl.classList.remove("hidden");
    statusEl.textContent = "Done.";
    tf.dispose([input, prediction]);
  } catch (error) {
    console.error("Prediction error:", error);
    statusEl.textContent = "Error: " + error.message + " (check console)";
    resultsEl.classList.add("hidden");
  } finally {
    predictBtn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  fileInput.addEventListener("change", handleFileChange);
  predictBtn.addEventListener("click", runPrediction);
  loadModel();
});



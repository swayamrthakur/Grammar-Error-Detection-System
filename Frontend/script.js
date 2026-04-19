// 🔗 Connect to Flask backend
async function getMLPrediction(text) {
  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    return await res.json();
  } catch (error) {
    console.error("ML API error:", error);
    return { prediction: "Error" };
  }
}


// 🧠 Main analysis function
async function analyzeText() {
  const text = document.getElementById("inputText").value;

  if (!text.trim()) return;

  let errors = 0;
  let words = text.split(/\s+/);
  let outputWords = [...words];

  // 🔥 Rules (grammar + spelling)
  const rules = [
    {
      wrong: "go",
      correct: "goes",
      type: "Grammar",
      message: "Use 'goes' with 'he/she'",
      condition: (w, i) => ["he", "she"].includes(words[i - 1]?.toLowerCase())
    },
    {
      wrong: "dont",
      correct: "doesn't",
      type: "Grammar",
      message: "Use 'doesn't' (missing apostrophe)"
    },
    {
      wrong: "was",
      correct: "were",
      type: "Grammar",
      message: "Use 'were' with plural subjects",
      condition: (w, i) => ["they", "we"].includes(words[i - 1]?.toLowerCase())
    },
    {
      wrong: "is",
      correct: "am",
      type: "Grammar",
      message: "Use 'am' with 'I'",
      condition: (w, i) => words[i - 1] === "I"
    },
    {
      wrong: "assignement",
      correct: "assignment",
      type: "Spelling",
      message: "Spelling mistake: assignment"
    },
    {
      wrong: "recieve",
      correct: "receive",
      type: "Spelling",
      message: "Spelling mistake: receive"
    }
  ];

  let htmlOutput = "";

  words.forEach((word, i) => {
    let corrected = word;
    let matchedRule = null;

    rules.forEach(rule => {
      if (
        word.toLowerCase() === rule.wrong &&
        (!rule.condition || rule.condition(word, i))
      ) {
        corrected = rule.correct;
        matchedRule = rule;
      }
    });

    if (matchedRule) {
      errors++;
      outputWords[i] = corrected;

      htmlOutput += `<span class="error" data-msg="${matchedRule.type}: ${matchedRule.message}">
        ${word}
      </span> `;
    } else {
      htmlOutput += word + " ";
    }
  });

  // 🖥️ Show highlighted output
  document.getElementById("outputText").innerHTML = htmlOutput;

  // 📊 Metrics
  document.getElementById("errors").innerText = errors;
  document.getElementById("words").innerText = words.length;

  // ✅ Corrected sentence
  const correctedText = outputWords.join(" ");

  let correctedDiv = document.getElementById("correctedText");

  if (!correctedDiv) {
    correctedDiv = document.createElement("div");
    correctedDiv.id = "correctedText";
    correctedDiv.style.marginTop = "10px";
    correctedDiv.style.color = "#4ade80";
    document.getElementById("outputText").after(correctedDiv);
  }

  correctedDiv.innerHTML = `<strong>Corrected:</strong> ${correctedText}`;

  // 🤖 ML Prediction (FINAL INTEGRATION)
  const mlResult = await getMLPrediction(text);
  document.getElementById("ml").innerText = mlResult.prediction;
}
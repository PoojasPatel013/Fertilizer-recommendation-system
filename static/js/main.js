document.addEventListener("DOMContentLoaded", () => {
  // Check if models are loaded
  fetch("/load_data")
    .then((response) => response.json())
    .then((data) => {
      console.log(data.status)
    })
    .catch((error) => {
      console.error("Error loading data:", error)
    })

  // Handle form submission
  const predictionForm = document.getElementById("predictionForm")
  const resultsContainer = document.getElementById("results")

  predictionForm.addEventListener("submit", (e) => {
    e.preventDefault()

    // Show loading indicator
    resultsContainer.innerHTML = `
            <div class="text-center py-5">
                <div class="spinner-border text-success" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mt-3">Analyzing data and generating recommendations...</p>
            </div>
        `

    // Get form data
    const formData = {
      temperature: document.getElementById("temperature").value,
      humidity: document.getElementById("humidity").value,
      moisture: document.getElementById("moisture").value,
      soil_type: document.getElementById("soil_type").value,
      nitrogen: document.getElementById("nitrogen").value,
      potassium: document.getElementById("potassium").value,
      phosphorous: document.getElementById("phosphorous").value,
    }

    // Send request to API
    fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.error) {
          resultsContainer.innerHTML = `
                    <div class="alert alert-danger">
                        ${data.error}
                    </div>
                `
        } else {
          displayResults(data)
        }
      })
      .catch((error) => {
        resultsContainer.innerHTML = `
                <div class="alert alert-danger">
                    An error occurred: ${error}
                </div>
            `
      })
  })

  // Function to display results
  function displayResults(data) {
    // Create HTML for top crops
    let topCropsHtml = ""
    data.top_crops.forEach((crop) => {
      const percentage = (crop.probability * 100).toFixed(1)
      topCropsHtml += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${crop.name}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-success" role="progressbar" style="width: ${percentage}%" aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </div>
            `
    })

    // Create HTML for top fertilizers
    let topFertilizersHtml = ""
    data.top_fertilizers.forEach((fertilizer) => {
      const percentage = (fertilizer.probability * 100).toFixed(1)
      topFertilizersHtml += `
                <div class="mb-2">
                    <div class="d-flex justify-content-between">
                        <span>${fertilizer.name}</span>
                        <span>${percentage}%</span>
                    </div>
                    <div class="progress">
                        <div class="progress-bar bg-success" role="progressbar" style="width: ${percentage}%" aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </div>
            `
    })

    // Determine sustainability class
    let sustainabilityClass = "bg-danger"
    if (data.sustainability_score >= 70) {
      sustainabilityClass = "bg-success"
    } else if (data.sustainability_score >= 40) {
      sustainabilityClass = "bg-warning"
    }

    // Determine productivity class
    let productivityClass = "bg-danger"
    if (data.productivity_score >= 70) {
      productivityClass = "bg-success"
    } else if (data.productivity_score >= 40) {
      productivityClass = "bg-warning"
    }

    // Display results
    resultsContainer.innerHTML = `
            <div class="result-item">
                <h5>Recommended Crop</h5>
                <p class="fs-4 fw-bold text-success">${data.crop}</p>
                <h6>Top Crop Recommendations:</h6>
                ${topCropsHtml}
            </div>
            
            <div class="result-item">
                <h5>Recommended Fertilizer</h5>
                <p class="fs-4 fw-bold text-success">${data.fertilizer}</p>
                <h6>Top Fertilizer Recommendations:</h6>
                ${topFertilizersHtml}
            </div>
            
            <div class="result-item">
                <h5>Sustainability Score</h5>
                <div class="d-flex justify-content-between">
                    <span>Score</span>
                    <span>${data.sustainability_score}/100</span>
                </div>
                <div class="progress mb-3">
                    <div class="progress-bar ${sustainabilityClass}" role="progressbar" style="width: ${data.sustainability_score}%" aria-valuenow="${data.sustainability_score}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                <p class="small text-muted">This score indicates how sustainable the recommended crop and fertilizer combination is for long-term soil health and environmental impact.</p>
            </div>
            
            <div class="result-item">
                <h5>Productivity Score</h5>
                <div class="d-flex justify-content-between">
                    <span>Score</span>
                    <span>${data.productivity_score}/100</span>
                </div>
                <div class="progress mb-3">
                    <div class="progress-bar ${productivityClass}" role="progressbar" style="width: ${data.productivity_score}%" aria-valuenow="${data.productivity_score}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>
                <p class="small text-muted">This score indicates the expected productivity and yield potential of the recommended crop with the suggested fertilizer.</p>
            </div>
            
            <div class="alert alert-info">
                <strong>Recommendation Summary:</strong> Based on your input parameters, ${data.crop} is the most suitable crop for your land, and ${data.fertilizer} is the recommended fertilizer. This combination offers a good balance between productivity and sustainability.
            </div>
        `
  }
})


const apiKey = "fa0a723aca0e4b59b57c32b84a1d3b28";

const newsContainer = document.getElementById("newsContainer");
const loader = document.getElementById("loader");

const countrySelect = document.getElementById("countrySelect");
const categorySelect = document.getElementById("categorySelect");
const searchInput = document.getElementById("searchInput");
const darkModeToggle = document.getElementById("darkModeToggle");

let darkMode = false;

// Format date for display
function formatDate(dateString) {
    const options = {
        year: "numeric", month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit"
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Fetch news
async function fetchNews(country = "in", category = "", query = "latest") {
    loader.style.display = "block";
    newsContainer.innerHTML = "";

    const today = new Date();
    const fromDate = new Date();
    fromDate.setDate(today.getDate() - 30);

    let url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&from=${fromDate.toISOString()}&to=${today.toISOString()}&sortBy=publishedAt&pageSize=100&apiKey=${apiKey}`;

    if (query === "latest" && country) {
        url = `https://newsapi.org/v2/top-headlines?country=${country}&category=${category}&pageSize=100&apiKey=${apiKey}`;
    }

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data.articles && data.articles.length > 0) {
            data.articles.sort((a, b) => new Date(b.publishedAt) - new Date(a.publishedAt));
            displayNews(data.articles);
        } else {
            newsContainer.innerHTML = "<p class='no-news'>No news found for the last 30 days.</p>";
        }
    } catch (error) {
        console.error("Error fetching news:", error);
        newsContainer.innerHTML = "<p class='error'>Error fetching news. Try again later.</p>";
    } finally {
        loader.style.display = "none";
    }
}

// Display news
function displayNews(articles) {
    newsContainer.innerHTML = "";

    articles.forEach(article => {
        const card = document.createElement("div");
        card.className = "news-card";

        card.innerHTML = `
            <img src="${article.urlToImage || 'placeholder.jpg'}" alt="News Image">
            <h3>${article.title || "No title available"}</h3>
            <p>${article.description || "No description available"}</p>
            <a href="${article.url}" target="_blank">Read More</a>
            <div class="publish-time">${formatDate(article.publishedAt)}</div>
        `;
        newsContainer.appendChild(card);
    });
}

// Event listeners
countrySelect.addEventListener("change", () => {
    fetchNews(countrySelect.value, categorySelect.value, searchInput.value || "latest");
});
categorySelect.addEventListener("change", () => {
    fetchNews(countrySelect.value, categorySelect.value, searchInput.value || "latest");
});
searchInput.addEventListener("input", () => {
    fetchNews(countrySelect.value, categorySelect.value, searchInput.value || "latest");
});
darkModeToggle.addEventListener("click", () => {
    darkMode = !darkMode;
    document.body.classList.toggle("dark-mode", darkMode);
    darkModeToggle.textContent = darkMode ? "☀️ Light Mode" : "🌙 Dark Mode";
});

// Initial fetch
fetchNews();


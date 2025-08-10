const apiKey = fa0a723aca0e4b59b57c32b84a1d3b28;

const countrySelect = document.getElementById("countrySelect");
const stateSelect = document.getElementById("stateSelect");
const categorySelect = document.getElementById("categorySelect");
const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const newsContainer = document.getElementById("newsContainer");

const states = {
    in: ["All States", "Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh"],
    us: ["All States", "California", "Texas", "Florida", "New York"],
    gb: ["All Regions", "England", "Scotland", "Wales", "Northern Ireland"]
};

function updateStateOptions() {
    stateSelect.innerHTML = "";
    states[countrySelect.value].forEach(state => {
        let option = document.createElement("option");
        option.value = (state === "All States" || state === "All Regions") ? "" : state;
        option.textContent = state;
        stateSelect.appendChild(option);
    });
}

async function fetchNews(query = "") {
    const country = countrySelect.value;
    const category = categorySelect.value;
    const state = stateSelect.value;

    let url = `https://newsapi.org/v2/top-headlines?country=${country}&apiKey=${apiKey}`;

    if (category) url += `&category=${category}`;

    // Combine search input query into q param
    if (query) {
        url += `&q=${encodeURIComponent(query)}`;
    }

    try {
        const res = await fetch(url);
        const data = await res.json();

        if (data.status !== "ok") {
            newsContainer.innerHTML = `<p>Error fetching news: ${data.message}</p>`;
            return;
        }

        // Filter articles client-side by state keyword
        const filteredArticles = filterByState(data.articles, state);
        displayNews(filteredArticles);
    } catch (error) {
        newsContainer.innerHTML = `<p>Error fetching news: ${error.message}</p>`;
    }
}

function filterByState(articles, state) {
    if (!state) return articles;
    const stateLower = state.toLowerCase();
    return articles.filter(article => {
        const text = ((article.title || "") + " " + (article.description || "") + " " + (article.content || "")).toLowerCase();
        return text.includes(stateLower);
    });
}

function displayNews(articles) {
    newsContainer.innerHTML = "";
    if (!articles || articles.length === 0) {
        newsContainer.innerHTML = "<p>No news found.</p>";
        return;
    }

    articles.forEach(article => {
        const card = document.createElement("div");
        card.classList.add("news-card");

        card.innerHTML = `
            <img src="${article.urlToImage || 'https://via.placeholder.com/300'}" alt="News Image">
            <div class="content">
                <h3>${article.title || "No Title"}</h3>
                <p>${article.description || ""}</p>
                <a href="${article.url}" target="_blank" rel="noopener noreferrer">Read more</a>
            </div>
        `;
        newsContainer.appendChild(card);
    });
}

// Live time update
setInterval(() => {
    document.getElementById("live-time").textContent = new Date().toLocaleString();
}, 1000);

// Event listeners
countrySelect.addEventListener("change", () => {
    updateStateOptions();
    fetchNews(searchInput.value);
});

stateSelect.addEventListener("change", () => fetchNews(searchInput.value));
categorySelect.addEventListener("change", () => fetchNews(searchInput.value));
searchBtn.addEventListener("click", () => fetchNews(searchInput.value));

// Initialize
updateStateOptions();
fetchNews();

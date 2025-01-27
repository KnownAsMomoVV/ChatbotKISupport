const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
app.use(cors());
app.use(express.json());

// Simple in-memory vector store
let vectorStore = [];

// Initialize with your documents (modify with your actual content)
const documents = [
  "Your first document text...",
  "Your second document text...",
  // Add more documents
];

// Initialize vector store with embeddings
(async () => {
  for (const text of documents) {
    const embedding = await getEmbedding(text);
    vectorStore.push({ text, embedding });
  }
  console.log("✅ Vector store initialized");
})();

// Get embeddings using Ollama
async function getEmbedding(text) {
  try {
    const response = await axios.post("http://127.0.0.1:11434/api/embeddings", {
      model: "nomic-embed-text", // Good embedding model for Ollama
      prompt: text,
    });
    return response.data.embedding;
  } catch (error) {
    console.error("Embedding error:", error);
    throw error;
  }
}

// Cosine similarity function
function cosineSimilarity(a, b) {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

app.post("/ask", async (req, res) => {
  try {
    const { question } = req.body;

    // 1. Get question embedding
    const queryEmbedding = await getEmbedding(question);

    // 2. Find similar documents
    const similarities = vectorStore.map(doc => ({
      text: doc.text,
      score: cosineSimilarity(queryEmbedding, doc.embedding)
    }));

    // Get top 3 results
    const context = similarities
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map(d => d.text)
      .join("\n---\n");

    // 3. Generate response
    const response = await axios.post("http://127.0.0.1:11434/api/generate", {
      model: "mistral",
      prompt: `[INST] Answer using this context:
${context}

Question: ${question}
Answer clearly in 1-3 short sentences.[/INST]`,
      stream: false
    });

    res.json({ answer: response.data.response });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));
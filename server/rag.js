const path = require('path');
const fs = require('fs').promises;
const pdfParse = require('pdf-parse');
const { Document } = require('@langchain/core/documents');

// Custom in-memory vector store (same as before)
class SimpleVectorStore {
  constructor() {
    this.vectors = [];
    this.documents = [];
  }

  async addVectors(vectors, documents) {
    this.vectors.push(...vectors);
    this.documents.push(...documents);
  }

  async similaritySearch(vector, k = 4) {
    const scores = this.vectors.map(v => this.cosineSimilarity(vector, v));
    const indices = scores
      .map((score, index) => ({ score, index }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(item => item.index);
    return indices.map(i => this.documents[i]);
  }

  cosineSimilarity(a, b) {
    const dot = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const normB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dot / (normA * normB);
  }
}

// Text splitter (same as before)
function splitText(text, chunkSize = 1000, overlap = 200) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = start + chunkSize;
    chunks.push(text.slice(start, end));
    start = end - overlap;
  }
  return chunks;
}

// New Ollama API helper
async function getOllamaEmbedding(text) {
  try {
    const response = await fetch('http://127.0.0.1:11434/api/embeddings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'mistral',
        prompt: text
      })
    });

    const data = await response.json();
    return data.embedding;
  } catch (error) {
    console.error('Embedding error:', error);
    throw error;
  }
}

async function loadFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();

  try {
    let text;
    if (ext === '.pdf') {
      const dataBuffer = await fs.readFile(filePath);
      ({ text } = await pdfParse(dataBuffer));
    } else if (ext === '.txt' || ext === '.md') {
      text = await fs.readFile(filePath, 'utf-8');
    } else {
      return [];
    }

    return splitText(text).map(content =>
      new Document({
        pageContent: content,
        metadata: { source: filePath }
      })
    );
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error.message);
    return [];
  }
}

async function setupVectorStore() {
  const vectorStore = new SimpleVectorStore();
  const knowledgePath = path.join(__dirname, '..', 'knowledge');
  const files = await fs.readdir(knowledgePath);

  for (const file of files) {
    const docs = await loadFile(path.join(knowledgePath, file));
    for (const doc of docs) {
      const embedding = await getOllamaEmbedding(doc.pageContent);
      await vectorStore.addVectors([embedding], [doc]);
    }
  }

  return vectorStore;
}

module.exports = { setupVectorStore };
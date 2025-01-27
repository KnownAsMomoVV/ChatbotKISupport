const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const { setupVectorStore, getEmbedding } = require('./rag');

const app = express();
app.use(cors());
app.use(express.json());

// Configuration paths
const DATA_DIR = path.join(__dirname, '..');
const PERSONA_FILE = path.join(DATA_DIR, 'knowledge', 'persona.txt');
const FAQ_FILE = path.join(DATA_DIR, 'knowledge', 'faq.txt');

let vectorStore;
let intentRecognizer;
let AI_PERSONA = {};

async function loadPersonaConfig() {
  try {
    const data = await fs.readFile(PERSONA_FILE, 'utf-8');
    const lines = data.split('\n').filter(line => line.trim() !== '');
    
    const persona = {
      name: "Default Assistant",
      style: "Helpful technical support",
      responseGuidelines: ["Provide clear, concise answers"]
    };
    
    let currentKey = '';
    
    lines.forEach(line => {
      if (line.startsWith('name:')) {
        persona.name = line.replace('name:', '').trim();
      } else if (line.startsWith('style:')) {
        persona.style = line.replace('style:', '').trim();
      } else if (line.startsWith('responseGuidelines:')) {
        currentKey = 'responseGuidelines';
        persona.responseGuidelines = [];
      } else if (currentKey === 'responseGuidelines' && line.startsWith('- ')) {
        persona.responseGuidelines.push(line.replace('- ', '').trim());
      }
    });

    // Ensure responseGuidelines exists
    persona.responseGuidelines = persona.responseGuidelines || [];
    return persona;
  } catch (error) {
    console.error('Error loading persona config:', error);
    return {
      name: "Default Assistant",
      style: "Helpful technical support",
      responseGuidelines: ["Provide clear, concise answers"]
    };
  }
}

async function initializeSystem() {
  try {
    // Load persona configuration
    AI_PERSONA = await loadPersonaConfig();
    
    // Initialize RAG system
    const ragSystem = await setupVectorStore(FAQ_FILE);
    vectorStore = ragSystem.vectorStore;
    intentRecognizer = ragSystem.intentRecognizer;
    
    console.log("✅ RAG system ready");
    console.log("Loaded personality:", AI_PERSONA.name);
  } catch (error) {
    console.error("❌ System initialization failed:", error);
    process.exit(1);
  }
}

// Initialize the system once
initializeSystem().catch(error => {
  console.error("❌ Critical initialization failed:", error);
  process.exit(1);
});

app.post('/ask', async (req, res) => {
  if (!vectorStore || !intentRecognizer) {
    return res.status(503).json({ error: "System initializing..." });
  }

  const { question } = req.body;
  if (!question) {
    return res.status(400).json({ error: "Question required" });
  }

  try {
    // 1. Intent Detection with Varied Responses
    const intent = await intentRecognizer.detectIntent(question);
    if (intent) {
      const intentDetails = intentRecognizer.intents.get(intent.name);
      const dynamicResponse = await generateDynamicAnswer(
        `System answer: ${intentDetails.answer}`,
        question,
        AI_PERSONA
      );
      
      return res.json({
        answer: dynamicResponse,
        sources: ["System Knowledge"],
        intent: intent.name
      });
    }

    // 2. Document-Based Response Generation
    const queryEmbedding = await getEmbedding(question);
    const results = vectorStore.similaritySearch(queryEmbedding, 5);
    
    if (results.length === 0) {
      return res.json({
        answer: await generateDynamicAnswer(
          "No relevant information found",
          question,
          AI_PERSONA
        ),
        sources: []
      });
    }

    // 3. Contextual Response Crafting
    const context = results
      .map(({ doc }, i) => `SOURCE ${i+1} (${doc.metadata.source}):\n${doc.pageContent}`)
      .join('\n\n');

    const finalAnswer = await generateDynamicAnswer(context, question, AI_PERSONA);
    
    res.json({
      answer: finalAnswer,
      sources: [...new Set(results.map(({ doc }) => doc.metadata.source))],
      intent: null
    });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ 
      answer: "⚓ Smooth sailing turned rough! Let's try that again...",
      error: error.message 
    });
  }
});

// AI Response Generator with Safety Checks
async function generateDynamicAnswer(context, question, persona) {
  try {
    // Safeguard against undefined guidelines
    const safePersona = {
      name: persona.name || "Assistant",
      style: persona.style || "Helpful AI",
      responseGuidelines: persona.responseGuidelines || ["Be helpful and concise"]
    };

    const guidelines = (safePersona.responseGuidelines || [])
      .map((g, i) => `${i+1}. ${g}`)
      .join('\n') || '1. Provide the best possible answer';

    const responsePrompt = `[INST] You are ${safePersona.name}, ${safePersona.style}.
Guidelines:
${guidelines}

Context Data:
${context}

User Question: ${question}

Create a helpful response that:
- Uses information EXACTLY as shown in context
- Varies structure from previous answers
- Adds personality without being repetitive
- Formats differently than last time
[/INST]`;

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'mistral',
        prompt: responsePrompt,
        stream: false,
        options: {
          temperature: 0.8,
          num_predict: 150,
          repeat_penalty: 1.2,
          top_k: 40,
          seed: Date.now()
        }
      })
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    
    if (!data?.response) {
      throw new Error('Invalid response format from Ollama');
    }

    return data.response
      .replace(/\[INST\].*\[\/INST\]/gs, '')
      .trim()
      .replace(/\n+/g, '\n');

  } catch (error) {
    console.error('Generation error:', error);
    return `🚨 Anchors aweigh! Let's try a different approach... (${error.message})`;
  }
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`🌊 Server running on port ${PORT}`));
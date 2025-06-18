# 📘 Course Title: _LLM Engineering: Master AI, Large Language Models & Agents_

## 🗓️ Lecture Title: _Lecture Topic or Date_

## 🧠 Learning Objectives

• **Build advanced Generative AI products** using cutting-edge models and frameworks.

• **Experiment with over 20 groundbreaking AI models**, including Frontier and Open-Source models.

• **Develop proficiency with platforms** like HuggingFace, LangChain, and Gradio.

• **Implement state-of-the-art techniques** such as RAG (Retrieval-Augmented Generation), QLoRA fine-tuning, and Agents.

• **Create real-world AI applications**, including:

• A multi-modal customer support assistant that interacts with text, sound, and images.

• An AI knowledge worker that can answer any question about a company based on its shared drive.

• An AI programmer that optimizes software, achieving performance improvements of over 60,000 times.

• An ecommerce application that accurately predicts prices of unseen products.

• **Transition from inference to training**, fine-tuning both Frontier and Open-Source models.

• **Deploy AI products to production** with polished user interfaces and advanced capabilities.

• **Level up your AI and LLM engineering skills** to be at the forefront of the industry.

## 📝 Key Concepts

### 1. Frontier LLM

- Definition: The term "frontier LLM" refers to large language models that are at the cutting edge of AI research and development.
  These models typically represent the most advanced, capable, and powerful systems available at a given time. Here's a breakdown of what characterizes a frontier LLM:

      Key Characteristics of Frontier LLMs:
      Scale: They are trained on massive datasets using billions or even trillions of parameters.
      Performance: They achieve state-of-the-art results on a wide range of benchmarks, including reasoning, coding, translation, and more.
      Capabilities: They often exhibit emergent behaviors—skills or abilities that were not explicitly programmed or expected.
      Safety and Alignment Focus: Due to their power, frontier LLMs are often the focus of intensive safety, alignment, and governance efforts.
      Innovation: They incorporate the latest architectural, training, and optimization techniques.

- Examples:

  - GPT-4 and successors by OpenAI
  - Gemini by Google DeepMind
  - Claude by Anthropic
  - Command R+ by Cohere
  - LLaMA 3 by Meta

  (detailed examples in the annex)

### 2. Open-source LLM

- Definition: A type of artificial intelligence language model whose architecture, model weights, training data (or sufficient description of it), and/or training code are publicly available under a license that permits use, modification, and distribution by anyone. Is less performant than a Frontier model.
- Example: Ollama by Magistral AI
- Notes:

  In this course, we will run Ollama model locally on our machine by running OllamaSetup exe,
download from official site: https://ollama.com/.
  
  Ollama has been trained with way less parameters than frontier models.
  In the range of 100 000 vs
  trillion parameters for a frontier model.
  
  **Try it**: ollama run llama3.2 or llama3.2:1b, as a faster alternative.

### 3. Skills and tools for AI development

Models:

- open-source (public)
- closed-source (proprietary technology)
- multi-modal
- architecture
- selecting

Tools:

- HuggingFace
- LangChain
- Gradio
- Weights & Biases
- Modal

Techniques:

- APIs
- Multi-shot prompting
- RAG
- Fine-tuning
- Agentization

### 4. Understanding Frontier models: GPT, Claude & Open Source LLMs

- Closed-Source Frontier:

  - GPT from OpenAI
  - Claude from Anthropic
  - Gemini from Google
  - Command R from Cohere
  - Perplexity

- Open-Source Frontier:
  - Llama from Meta (Ollama named after)
  - Mixtral from Mistral
  - Qwen from Alibaba Cloud
  - Gemma from Google - smaller
  - Phi from Microsoft - smaller

**Three ways to use models**:

1. Chat interfaces (like ChatGPT)
2. Cloud APIs (LLM API)
   Frameworks like LangChain

   Managed AI cloud services:

   - Amazon Bedrock
   - Google Vertex
   - Azure ML

Here you are connecting with a provider like Amazon, Google Azure and they are running the model on their cloud.
Offering you to choose between closed-source or open-source models.

3. Direct inference
   With the HuggingFace
   Transformers library
   With Ollama to run locally

- Using google collab -> running in the cloud for power
- Using Ollama to run locally (optimized using C++ - CPP)

### 5. Key knowledge

- RAG - knowledge store to apply to LLMs
- Fine tuning Frontier model
- Fine tuning an open source model
- Gradio: build a nice sharp user interface very quickly
- Agents to solve a real world case scenario

## Installation

- Install Python version 3.11.9 (not latest), because this version supports better data science dependencies (as discussed in course)

Complete installation setup for PC:
https://github.com/ed-donner/llm_engineering/blob/main/README.md

Chose not to install Anaconda (heavy data science env.) and instead intstall a lightweight virtual environment using python.
I followed Part 2B - Alternative to Part 2.

**Recurring commands**

From within the llm_engineering folder in a Windows Terminal, run:

- Activate the environment:

  - venv\Scripts\activate

- Start jupyter lab:
  - jupyter lab

Setting up OpenAI API for LLM

- Setting up API keys with OpenAI
- Adding to the project folder an .env file with a OpenAI API generated key
- Load the configuration into the Python script using a package like python-dotenv

## Implementing text summarization using OpenAI's GPT-4 and BeautifulSoup

Goal: calling the cloud API of a Frontier model (a leading model at the frontier of AI)

1. Using OpenAI completion API with GPT-4o-mini
2. Using summarization - a classic Gen AI use case to make a summary (web scraping)

**Types of prompts**

Models like GPT4o have been trained to receive instructions in a particular way.
They expect to receive:

- A system prompt that tells them what task they are performing and what tone they should use

- A user prompt -- the conversation starter that they should reply to

Frontier models trained in two different ways, so they are:

- system prompt

  - provides context, tone

- user prompt
  - conversation itself

## 🧮 Useful commands

- Run the following pwsh commands to activate and run the Jupyter Lab environment (once installation setup is done):

  - venv\Scripts\activate
  - jupyter lab
    > (python application exe)

- Run _ollama serve_ if Ollama is down.

## Annex

**Examples of Frontier Models**:

🧠 1. OpenAI
These models are designed for cutting-edge reasoning, multimodal interaction, and general-purpose intelligence:
GPT-4 – Flagship model known for strong reasoning and language capabilities.
GPT-4 Turbo – Optimized version of GPT-4 for speed and cost.
GPT-4o – Multimodal model (text, vision, audio) with real-time capabilities.
o1-preview – Successor to GPT-4o in reasoning tasks; excels in math, science, and code 1.

🧠 2. Google DeepMind
DeepMind’s Gemini series is their frontier model family:
Gemini 1.5 Pro – High-end model with strong reasoning and multimodal capabilities.
Gemini 1.5 Flash – Optimized for speed and efficiency.
Gemini 1.0 Ultra – Earlier frontier model with advanced capabilities2.

🧠 3. Anthropic
Anthropic’s Claude 3 family includes:
Claude 3 Opus – Their most powerful and capable model; explicitly a frontier model.
Claude 3 Sonnet – Mid-tier model, not explicitly frontier but still highly capable.
Claude 3 Haiku – Lightweight, fast model for simpler tasks.

🧠 4. Cohere
Cohere focuses on enterprise and retrieval-augmented generation (RAG), with:
Command R+ – Their most advanced model, optimized for RAG and enterprise use cases.
Command R – Earlier version, still powerful for structured tasks.
While not always labeled as "frontier" in the same way as OpenAI or DeepMind models, Command R+ is Cohere’s top-tier offering.

🧠 5. Meta (Facebook AI)
Meta’s LLaMA (Large Language Model Meta AI) series includes:
LLaMA 3 70B – Their most capable open-weight model, considered frontier-level in open-source AI.
LLaMA 2 70B – Previous generation, still widely used in research and industry.
Meta positions these models as open frontier models, especially for academic and developer communities.

**Example of generated output:**

Generated from Ollama in **day2 EXERCISE**:

Hands-on LLM task: comparing OpenAI and Ollama for text summarization
Pour le même exercise, OpenAI donne un résumé plus complet que Ollama...

Generative AI has a wide range of business applications across various industries. Here are some examples:

1. **Virtual Reality (VR) and Augmented Reality (AR)**: Generative AI can be used to create immersive VR and AR experiences for training, marketing, and customer engagement.
2. **Content Creation**: Generative AI can generate high-quality content such as text articles, social media posts, product descriptions, and even entire websites. This can save businesses time and resources while improving the quality of their content.
3. **Chatbots and Customer Service**: Generative AI-powered chatbots can provide 24/7 customer support, answering frequently asked questions, routing complex issues to human agents, and even generating responses to common customer inquiries.
4. **Predictive Maintenance**: Generative AI can analyze sensor data from machines and predict when maintenance is required, reducing downtime and increasing overall efficiency.
5. **Supply Chain Optimization**: Generative AI can help optimize supply chain operations by predicting demand, identifying potential bottlenecks, and suggesting alternative routes or transportation methods.
6. **Cybersecurity**: Generative AI can be used to analyze vast amounts of data to identify potential security threats, detect anomalies, and provide real-time alerts for potential breaches.
7. **Product Development**: Generative AI can help design new products by analyzing market trends, user behavior, and competitor offerings. This can lead to the creation of innovative products that meet customer needs better than traditional methods.
8. **Education**: Generative AI can create personalized learning plans, adaptive assessments, and even entire courses for students based on their individual needs and progress.
9. **Marketing Automation**: Generative AI can help automate marketing processes such as content creation, lead generation, and campaign optimization, allowing businesses to save time and resources while improving their marketing effectiveness.
10. **Data Analytics**: Generative AI can help analyze large datasets by identifying patterns, trends, and correlations that may not be apparent through manual analysis.
11. **Medical Imaging Analysis**: Generative AI can help medical professionals analyze medical images such as X-rays, MRIs, and CT scans to identify potential health issues earlier and more accurately than traditional methods.
12. **Financial Modeling**: Generative AI can help financial models create realistic scenarios, predict market trends, and optimize investment strategies.

These are just a few examples of the many business applications of generative AI. As the technology continues to evolve, we can expect to see even more innovative uses in various industries.

Also trying the amazing reasoning model DeepSeek (of Ollama)
Here we use the version of DeepSeek-reasoner that's been distilled to 1.5B.
This is actually a 1.5B variant of Qwen that has been fine-tuned using synethic data generated by Deepseek R1.
Other sizes of DeepSeek are here all the way up to the full 671B parameter version, which would use up 404GB of your drive and is far too large for most!

Try it!

## 📊 Diagrams / Visuals

> _(Insert image or diagram here if needed)_  
> Example: `!Alt text`

---

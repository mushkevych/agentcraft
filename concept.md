# Bioengineering & Drug Discovery Assistant (xi)

AgentCraft demonstrator project solves a realistic business problem while showcasing agentic architecture with LangChain & LangGraph.

**Business Problem**:
Pharmaceutical companies spend billions of dollars on drug research, but finding relevant papers, clinical trials, and previous research is a time-consuming process.

## **Agentic Solution (LangGraph + LangChain)**

  * Request Router Node: Detects user intent (e.g., "Find related studies on CRISPR gene editing").
  * PostgreSQL DB Node: Queries a local ChEMBL database of bioactivity data on drugs and small molecules.
  * Web Search Node (Tavily API): Fetches the latest bioengineering research from journals like PubMed, arXiv, etc.
  * Summarization & Looping Node (ReAct): Extracts key insights, highlights contradictions, and loops if additional information is needed.

### **Use Case**
A biotech company wants to quickly evaluate new advancements in gene therapy, automatically surfacing key insights from research papers, patents, and news articles.

## **Expected list of questions** 
for the Bioengineering & Drug Discovery Assistant:

### ChEMBL DB node capability

#### **Drug & Compound Information**
- **"What are the properties of Aspirin?"**
- **"Which ChEMBL ID corresponds to Ibuprofen?"**
- **"What is the molecular weight of Paracetamol?"**
- **"What are the chemical representations (SMILES, InChI) of Penicillin?"**

#### **Bioactivity & Mechanism of Action**
- **"How does Metformin interact with its target?"**
- **"What is the IC50 value of Drug X against Target Y?"**
- **"What are the known bioactivities of Hydroxychloroquine?"**

#### **Target & Protein Interactions**
- **"Which proteins does Drug X bind to?"**
- **"Which small molecules inhibit EGFR (Epidermal Growth Factor Receptor)?"**
- **"What is the UniProt ID for the target of Remdesivir?"**

#### **Drug Similarity & Structural Queries**
- **"Find drugs similar to Atorvastatin."**
- **"What compounds have a similar molecular fingerprint to Ibuprofen?"**
- **"Which molecules share the same target as Drug X?"**

#### **Clinical & Regulatory Information**
- **"What phase of clinical trials is Drug X in?"**
- **"Which drugs are FDA-approved for treating Disease Y?"**
- **"What drugs were withdrawn due to toxicity?"**

### Internet search node capability
While ChEMBL provides a **structured, static database**, xi can use **Tavily** for **real-time** insights, such as:

#### **Latest Drug Research & Discoveries**
- **"What are the latest research findings on CRISPR-based gene therapy?"**
- **"What new COVID-19 antivirals have been approved?"**
- **"What recent studies link Metformin to longevity?"**

#### **Clinical Trials & Regulatory Updates**
- **"Are there new clinical trials for Alzheimer's disease drugs?"**
- **"Has Drug X been approved by the FDA in 2024?"**
- **"What are the latest safety concerns about GLP-1 receptor agonists?"**

#### **Patent & Intellectual Property Research**
- **"Who holds the patent for Ozempic?"**
- **"Which companies are leading in gene therapy patents?"**
- **"Are there any new biosimilar versions of Humira?"**

#### **Scientific Literature & Research Papers**
- **"What does PubMed say about the efficacy of Drug X for cancer treatment?"**
- **"Are there any new studies on the mechanism of action of Drug Y?"**
- **"Summarize the latest Nature paper on mRNA vaccine technology."**

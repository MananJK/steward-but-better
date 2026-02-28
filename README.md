# Steward But Better

A professional FIA steward decision-making assistant powered by Mistral AI.

## System Architecture

### Mistral OCR Pipeline
1. **Document Ingestion**: PDF rulebooks are processed through Mistral's OCR system
2. **Text Extraction**: Optical character recognition converts scanned documents to searchable text
3. **Structured Storage**: Processed rules are stored in `processed_rules/` directory

### RAG (Retrieval-Augmented Generation) Pipeline
1. **Vector Indexing**: Rules are embedded into vector space using Mistral's embedding models
2. **Semantic Search**: Incident queries retrieve relevant rule sections via vector similarity
3. **Context-Augmented Reasoning**: Retrieved rules provide context for LLM-based decision making
4. **Judicial Output**: Structured decisions with rule citations and reasoning

## Key Components

- `src/brain/steward_agent.py` - Core decision-making agent
- `src/brain/vector_index.py` - RAG vector database
- `src/ingestion/` - OCR processing pipeline
- `src/telemetry/` - Incident monitoring and evaluation
- `src/vision/` - Video analysis tools

## Rulebooks

See `JUDICIAL_LOG.md` for complete list of ingested FIA regulations.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/brain/steward_agent.py --incident <incident_file>
```

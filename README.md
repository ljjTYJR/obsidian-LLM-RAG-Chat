# Gemini RAG Plugin

An Obsidian plugin that enables AI-powered search and chat functionality using Google Gemini and RAG (Retrieval-Augmented Generation) technology.

## Features

- **Smart Search**: Find relevant content across your vault using semantic similarity
- **AI Chat**: Chat with your notes using Gemini AI with contextual understanding  
- **Embeddings Database**: Automatically creates and saves vector embeddings of your markdown files
- **Multiple Interfaces**: Ribbon icons, commands, and modal interfaces for easy access

## Usage

### Setup
1. Install the plugin
2. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
3. Open plugin settings and enter your API key
4. Run "Rebuild Embeddings Database" command to process your vault

### Search & Chat
- Click the brain icon (🧠) in the ribbon for RAG search
- Click the chat icon (💬) in the ribbon to open the chat sidebar
- Use Command Palette: "Gemini RAG: Open RAG Search" or "Gemini RAG: Open RAG Chat"
- Select text and run "Gemini RAG: Query with RAG Context" to query selected content

### Commands
- **Open RAG Search**: Search your vault with AI-powered similarity matching
- **Open RAG Chat**: Start a conversational chat session with your notes
- **Rebuild Embeddings Database**: Reprocess all markdown files (run after adding new content)
- **Query with RAG Context**: Query selected text with relevant vault context

## Settings

Configure embedding models, chunk sizes, API settings, and retrieval parameters in the plugin settings tab.
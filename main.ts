import { Editor, MarkdownView, Notice, Plugin, TFile, WorkspaceLeaf } from 'obsidian';
import { promises as fs } from 'fs';
import { join } from 'path';
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document } from '@langchain/core/documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { CHAT_VIEW_TYPE, GeminiRAGSettings, DEFAULT_SETTINGS, DocumentChunk } from './src/types';
import { RAGSearchModal } from './src/rag-search-modal';
import { ChatView } from './src/chat-view';
import { GeminiRAGSettingTab } from './src/settings-tab';

export default class GeminiRAGPlugin extends Plugin {
	settings: GeminiRAGSettings;
	embeddings: GoogleGenerativeAIEmbeddings | null = null;
	llm: ChatGoogleGenerativeAI | null = null;
	vectorStore: MemoryVectorStore | null = null;
	ragChain: any = null;
	statusBarItem: HTMLElement;
	textSplitter: RecursiveCharacterTextSplitter;
	embeddingCache: Map<string, any> = new Map(); // For compatibility
	embeddingsPath: string;

	async onload() {
		await this.loadSettings();
		await this.forceRefreshSettings();

		// Set embeddings path to plugin directory
		this.embeddingsPath = join(this.app.vault.configDir, 'plugins', 'obsidian-llmtalk', 'embeddings.json');

		// Register the chat view
		this.registerView(
			CHAT_VIEW_TYPE,
			(leaf) => new ChatView(leaf, this)
		);

		// Add status bar item
		this.statusBarItem = this.addStatusBarItem();
		this.initializeGemini();

		// Add ribbon icon for search
		const ribbonIconEl = this.addRibbonIcon('brain-circuit', 'Gemini RAG Search', (evt: MouseEvent) => {
			this.openRAGSearchModal();
		});
		ribbonIconEl.addClass('gemini-rag-ribbon-class');

		// Add ribbon icon for chat sidebar
		const chatRibbonIconEl = this.addRibbonIcon('message-circle', 'Gemini RAG Chat', (evt: MouseEvent) => {
			this.activateChatView();
		});
		chatRibbonIconEl.addClass('gemini-chat-ribbon-class');

		// Add commands
		this.addCommand({
			id: 'gemini-rag-search',
			name: 'Open RAG Search',
			callback: () => {
				this.openRAGSearchModal();
			}
		});

		this.addCommand({
			id: 'gemini-rag-chat',
			name: 'Open RAG Chat',
			callback: () => {
				this.activateChatView();
			}
		});

		this.addCommand({
			id: 'gemini-rag-rebuild-embeddings',
			name: 'Rebuild Embeddings Database',
			callback: () => {
				this.rebuildEmbeddings();
			}
		});

		this.addCommand({
			id: 'gemini-rag-query',
			name: 'Query with RAG Context',
			editorCallback: (editor: Editor, view: MarkdownView) => {
				const selection = editor.getSelection();
				if (selection) {
					this.queryWithRAG(selection);
				} else {
					new Notice('Please select text to query');
				}
			}
		});

		// Add settings tab
		this.addSettingTab(new GeminiRAGSettingTab(this.app, this));

	}

	onunload() {
		// Cleanup if needed
	}

	initializeGemini() {
		if (this.settings.apiKey) {
			this.embeddings = new GoogleGenerativeAIEmbeddings({
				apiKey: this.settings.apiKey,
				model: this.settings.embeddingModel,
				maxRetries: 3,
				maxConcurrency: 1
			});
			this.llm = new ChatGoogleGenerativeAI({
				apiKey: this.settings.apiKey,
				model: this.settings.generativeModel,
				maxRetries: 3,
				temperature: 0.1,
				maxOutputTokens: 2048
			});
			this.textSplitter = new RecursiveCharacterTextSplitter({
				chunkSize: this.settings.chunkSize,
				chunkOverlap: this.settings.chunkOverlap
			});
			this.updateStatusBar('Gemini Ready');

			// Try to load existing embeddings
			this.loadExistingEmbeddings();
		} else {
			this.updateStatusBar('API Key Required');
		}
	}

	updateStatusBar(text: string) {
		if (this.statusBarItem) {
			this.statusBarItem.setText(`Gemini RAG: ${text}`);
		}
	}

	async loadExistingEmbeddings() {
		const loaded = await this.loadEmbeddings();
		if (!loaded) {
			this.updateStatusBar('Gemini Ready - No embeddings');
		}
	}

	async openRAGSearchModal() {
		if (!this.embeddings || !this.llm) {
			new Notice('Please configure your Gemini API key in settings');
			return;
		}
		new RAGSearchModal(this.app, this).open();
	}


	async activateChatView() {
		if (!this.embeddings || !this.llm) {
			new Notice('Please configure your Gemini API key in settings');
			return;
		}

		const { workspace } = this.app;

		let leaf: WorkspaceLeaf | null = null;
		const leaves = workspace.getLeavesOfType(CHAT_VIEW_TYPE);

		if (leaves.length > 0) {
			// A chat view is already open
			leaf = leaves[0];
		} else {
			// No chat view open, create a new one
			leaf = workspace.getRightLeaf(false);
			if (leaf) {
				await leaf.setViewState({ type: CHAT_VIEW_TYPE, active: true });
			}
		}

		// Focus the chat view
		if (leaf) {
			workspace.revealLeaf(leaf);
		}
	}

	async rebuildEmbeddings() {
		if (!this.embeddings || !this.llm) {
			new Notice('Please configure your Gemini API key in settings');
			return;
		}

		this.updateStatusBar('Building embeddings...');
		new Notice('Building embeddings database. This may take a while...');

		try {
			const markdownFiles = this.app.vault.getMarkdownFiles();
			const documents: Document[] = [];

			for (const file of markdownFiles) {
				const content = await this.app.vault.read(file);
				const chunks = await this.textSplitter.splitText(content);

				for (const chunk of chunks) {
					documents.push(new Document({
						pageContent: chunk,
						metadata: { source: file.path, fileName: file.name }
					}));
				}
			}

			this.vectorStore = await MemoryVectorStore.fromDocuments(documents, this.embeddings);

			// Build RAG chain
			await this.buildRAGChain();

			// Save embeddings to disk
			await this.saveEmbeddings();

			this.updateStatusBar(`Embeddings ready (${documents.length} chunks)`);
			new Notice(`Embeddings built successfully! Processed ${markdownFiles.length} files and saved to disk.`);
		} catch (error) {
			console.error('Error building embeddings:', error);
			new Notice('Error building embeddings. Check console for details.');
			this.updateStatusBar('Error building embeddings');
		}
	}

	async queryWithRAG(query: string): Promise<string> {
		if (!this.ragChain) {
			new Notice('Please build embeddings first');
			return 'Please build embeddings first using the "Rebuild Embeddings Database" command.';
		}

		return await this.retryWithFallback(async () => {
			this.updateStatusBar('Querying...');
			const result = await this.ragChain.invoke({ input: query });
			this.updateStatusBar('Ready');
			return result.answer;
		});
	}

	async retryWithFallback<T>(operation: () => Promise<T>): Promise<T> {
		const fallbackModels = ['gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-1.0-pro'];

		for (let attempt = 0; attempt < 3; attempt++) {
			try {
				return await operation();
			} catch (error: any) {
				console.error(`Attempt ${attempt + 1} failed:`, error);

				if (error.message?.includes('503') || error.message?.includes('Service Unavailable')) {
					this.updateStatusBar(`Retrying... (${attempt + 1}/3)`);

					// Try different model on second attempt
					if (attempt === 1 && this.llm) {
						const fallbackModel = fallbackModels[attempt % fallbackModels.length];
						console.log(`Switching to fallback model: ${fallbackModel}`);
						this.llm = new ChatGoogleGenerativeAI({
							apiKey: this.settings.apiKey,
							model: fallbackModel,
							maxRetries: 2,
							temperature: 0.1,
							maxOutputTokens: 2048
						});
						// Rebuild chain with new model
						if (this.vectorStore) {
							await this.rebuildChainWithNewModel();
						}
					}

					// Wait before retry with exponential backoff
					await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 2000));
					continue;
				} else {
					// Non-503 errors, handle immediately
					let errorMessage = 'An error occurred while querying.';
					if (error.message?.includes('401') || error.message?.includes('API key')) {
						errorMessage = 'Invalid API key. Please check your settings.';
						this.updateStatusBar('Invalid API Key');
					} else if (error.message?.includes('429') || error.message?.includes('quota')) {
						errorMessage = 'Rate limit exceeded. Please wait before trying again.';
						this.updateStatusBar('Rate Limited');
					} else {
						this.updateStatusBar('Error');
					}

					new Notice(errorMessage);
					throw new Error(errorMessage);
				}
			}
		}

		// All attempts failed
		this.updateStatusBar('Service Unavailable');
		const errorMessage = 'Google AI service is unavailable after multiple attempts. Please try again later.';
		new Notice(errorMessage);
		throw new Error(errorMessage);
	}

	async rebuildChainWithNewModel() {
		await this.buildRAGChain();
	}

	async searchSimilarChunks(query: string): Promise<DocumentChunk[]> {
		if (!this.vectorStore) return [];
		const docsWithScores = await this.vectorStore.similaritySearchWithScore(query, this.settings.maxResults);
		return docsWithScores.map(([doc, score]) => ({
			content: doc.pageContent,
			filePath: doc.metadata.source || '',
			fileName: doc.metadata.fileName || '',
			similarity: Math.round((1 - score) * 100) / 100 // Convert distance to similarity score
		}));
	}

	getTotalChunks(): number {
		return this.vectorStore?.memoryVectors?.length || 0;
	}

	async saveEmbeddings() {
		if (!this.vectorStore) return;

		try {
			const vectorData = {
				memoryVectors: this.vectorStore.memoryVectors,
				timestamp: Date.now()
			};

			await fs.writeFile(this.embeddingsPath, JSON.stringify(vectorData, null, 2));
			console.log('Embeddings saved to disk');
		} catch (error) {
			console.error('Error saving embeddings:', error);
		}
	}

	async loadEmbeddings(): Promise<boolean> {
		try {
			const data = await fs.readFile(this.embeddingsPath, 'utf-8');
			const vectorData = JSON.parse(data);

			if (!this.embeddings) return false;

			// Create new vector store with saved data
			this.vectorStore = new MemoryVectorStore(this.embeddings);
			this.vectorStore.memoryVectors = vectorData.memoryVectors;

			// Rebuild the RAG chain with loaded embeddings
			await this.buildRAGChain();

			const chunkCount = this.getTotalChunks();
			this.updateStatusBar(`Embeddings loaded (${chunkCount} chunks)`);
			console.log(`Embeddings loaded from disk: ${chunkCount} chunks`);
			return true;
		} catch (error) {
			console.log('No existing embeddings found or error loading:', error.message);
			return false;
		}
	}

	async buildRAGChain() {
		if (!this.llm || !this.vectorStore) return;

		const prompt = ChatPromptTemplate.fromTemplate(`
			Answer the following question based only on the provided context:

			<context>
			{context}
			</context>

			Question: {input}`);

		const documentChain = await createStuffDocumentsChain({
			llm: this.llm,
			prompt,
		});

		this.ragChain = await createRetrievalChain({
			retriever: this.vectorStore.asRetriever(),
			combineDocsChain: documentChain,
		});
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
		this.initializeGemini();
	}

	async forceRefreshSettings() {
		// Force reload from defaults if settings are corrupted or outdated
		const savedData = await this.loadData();
		if (!savedData || !savedData.generativeModel) {
			this.settings = { ...DEFAULT_SETTINGS };
			await this.saveSettings();
			console.log('Settings reset to defaults');
		}
	}
}

import { Editor, MarkdownView, Notice, Plugin, TFile, WorkspaceLeaf } from 'obsidian';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { CHAT_VIEW_TYPE, GeminiRAGSettings, DEFAULT_SETTINGS, DocumentChunk, EmbeddingData } from './types';
import { RAGSearchModal } from './rag-search-modal';
import { ChatView } from './chat-view';
import { GeminiRAGSettingTab } from './settings-tab';
import { EmbeddingUtils } from './embedding-utils';

export default class GeminiRAGPlugin extends Plugin {
	settings: GeminiRAGSettings;
	genAI: GoogleGenerativeAI | null = null;
	embeddingCache: Map<string, EmbeddingData> = new Map();
	statusBarItem: HTMLElement;

	async onload() {
		await this.loadSettings();

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

		// Load existing embeddings
		await this.loadEmbeddingsFromCache();
	}

	onunload() {
		this.saveEmbeddingsToCache();
	}

	initializeGemini() {
		if (this.settings.apiKey) {
			this.genAI = new GoogleGenerativeAI(this.settings.apiKey);
			this.updateStatusBar('Gemini Ready');
		} else {
			this.updateStatusBar('API Key Required');
		}
	}

	updateStatusBar(text: string) {
		if (this.statusBarItem) {
			this.statusBarItem.setText(`Gemini RAG: ${text}`);
		}
	}

	async openRAGSearchModal() {
		if (!this.genAI) {
			new Notice('Please configure your Gemini API key in settings');
			return;
		}
		new RAGSearchModal(this.app, this).open();
	}


	async activateChatView() {
		if (!this.genAI) {
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
		if (!this.genAI) {
			new Notice('Please configure your Gemini API key in settings');
			return;
		}

		this.updateStatusBar('Building embeddings...');
		new Notice('Building embeddings database. This may take a while...');

		try {
			const markdownFiles = this.app.vault.getMarkdownFiles();
			let processedFiles = 0;

			for (const file of markdownFiles) {
				await this.processFileForEmbeddings(file);
				processedFiles++;
				this.updateStatusBar(`Processing: ${processedFiles}/${markdownFiles.length}`);
			}

			await this.saveEmbeddingsToCache();
			this.updateStatusBar(`Embeddings ready (${this.getTotalChunks()} chunks)`);
			new Notice(`Embeddings built successfully! Processed ${processedFiles} files.`);
		} catch (error) {
			console.error('Error building embeddings:', error);
			new Notice('Error building embeddings. Check console for details.');
			this.updateStatusBar('Error building embeddings');
		}
	}

	async processFileForEmbeddings(file: TFile): Promise<void> {
		if (!this.genAI) return;
		
		const embeddingData = await EmbeddingUtils.processFileForEmbeddings(
			file,
			this.app.vault,
			this.genAI,
			this.settings
		);
		
		if (embeddingData) {
			this.embeddingCache.set(file.path, embeddingData);
		}
	}


	async getEmbedding(text: string): Promise<number[] | null> {
		return EmbeddingUtils.getEmbedding(text, this.genAI!, this.settings);
	}

	async searchSimilarChunks(query: string): Promise<DocumentChunk[]> {
		if (!this.genAI) return [];

		const queryEmbedding = await this.getEmbedding(query);
		if (!queryEmbedding) return [];

		return EmbeddingUtils.searchSimilarChunks(query, queryEmbedding, this.embeddingCache, this.settings);
	}


	async queryWithRAG(query: string): Promise<string> {
		if (!this.genAI) {
			throw new Error('Gemini not initialized');
		}

		const relevantChunks = await this.searchSimilarChunks(query);

		if (relevantChunks.length === 0) {
			new Notice('No relevant content found in your vault');
			return 'No relevant content found in your vault for this query.';
		}

		const context = relevantChunks
			.map(chunk => `From "${chunk.fileName}":\n${chunk.content}`)
			.join('\n\n---\n\n');

		const prompt = `Based on the following content from the user's Obsidian vault, please answer their question:

Context:
${context}

Question: ${query}

Please provide a comprehensive answer based on the provided context. If the context doesn't contain enough information to fully answer the question, mention that and provide what information is available.`;

		try {
			const model = this.genAI.getGenerativeModel({ model: this.settings.generativeModel });
			const result = await model.generateContent(prompt);
			const response = result.response;
			return response.text();
		} catch (error) {
			console.error('Error querying Gemini:', error);
			throw error;
		}
	}

	getTotalChunks(): number {
		let total = 0;
		for (const embeddingData of this.embeddingCache.values()) {
			total += embeddingData.chunks.length;
		}
		return total;
	}

	async loadEmbeddingsFromCache(): Promise<void> {
		try {
			const cacheData = await this.loadData();
			if (cacheData && cacheData.embeddingCache) {
				this.embeddingCache = new Map(Object.entries(cacheData.embeddingCache));
				this.updateStatusBar(`Loaded (${this.getTotalChunks()} chunks)`);
			}
		} catch (error) {
			console.error('Error loading embeddings cache:', error);
		}
	}

	async saveEmbeddingsToCache(): Promise<void> {
		try {
			const cacheData = {
				embeddingCache: Object.fromEntries(this.embeddingCache)
			};
			await this.saveData(cacheData);
		} catch (error) {
			console.error('Error saving embeddings cache:', error);
		}
	}

	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
		this.initializeGemini();
	}
}

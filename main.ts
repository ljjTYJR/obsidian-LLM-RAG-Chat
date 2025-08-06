import { App, Editor, MarkdownView, Modal, Notice, Plugin, PluginSettingTab, Setting, TFile, ItemView, WorkspaceLeaf, MarkdownRenderer } from 'obsidian';
import { GoogleGenerativeAI } from '@google/generative-ai';

export const CHAT_VIEW_TYPE = "gemini-rag-chat-view";

interface GeminiRAGSettings {
	apiKey: string;
	embeddingModel: string;
	generativeModel: string;
	maxResults: number;
	similarityThreshold: number;
	chunkSize: number;
	chunkOverlap: number;
}

const DEFAULT_SETTINGS: GeminiRAGSettings = {
	apiKey: '',
	embeddingModel: 'text-embedding-004',
	generativeModel: 'gemini-1.5-flash',
	maxResults: 5,
	similarityThreshold: 0.7,
	chunkSize: 1000,
	chunkOverlap: 200
}

interface DocumentChunk {
	content: string;
	filePath: string;
	fileName: string;
	embedding?: number[];
	similarity?: number;
}

interface ChatMessage {
	role: 'user' | 'assistant';
	content: string;
	timestamp: number;
	sources?: DocumentChunk[];
}

interface EmbeddingData {
	chunks: DocumentChunk[];
	lastUpdated: number;
}

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
		try {
			const content = await this.app.vault.read(file);
			const chunks = this.chunkText(content, this.settings.chunkSize, this.settings.chunkOverlap);

			const documentChunks: DocumentChunk[] = [];

			for (const chunk of chunks) {
				if (chunk.trim().length < 50) continue; // Skip very short chunks

				const embedding = await this.getEmbedding(chunk);
				if (embedding) {
					documentChunks.push({
						content: chunk,
						filePath: file.path,
						fileName: file.name,
						embedding: embedding
					});
				}
			}

			this.embeddingCache.set(file.path, {
				chunks: documentChunks,
				lastUpdated: Date.now()
			});
		} catch (error) {
			console.error(`Error processing file ${file.path}:`, error);
		}
	}

	chunkText(text: string, chunkSize: number, overlap: number): string[] {
		const chunks: string[] = [];
		let start = 0;

		while (start < text.length) {
			const end = Math.min(start + chunkSize, text.length);
			let chunk = text.slice(start, end);

			// Try to break at sentence boundaries
			if (end < text.length) {
				const lastPeriod = chunk.lastIndexOf('.');
				const lastNewline = chunk.lastIndexOf('\n');
				const breakPoint = Math.max(lastPeriod, lastNewline);

				if (breakPoint > start + chunkSize * 0.7) {
					chunk = text.slice(start, breakPoint + 1);
					start = breakPoint + 1 - overlap;
				} else {
					start = end - overlap;
				}
			} else {
				start = end;
			}

			chunks.push(chunk.trim());
		}

		return chunks.filter(chunk => chunk.length > 0);
	}

	async getEmbedding(text: string): Promise<number[] | null> {
		if (!this.genAI) return null;

		try {
			const model = this.genAI.getGenerativeModel({ model: this.settings.embeddingModel });
			const result = await model.embedContent(text);
			return result.embedding.values;
		} catch (error) {
			console.error('Error getting embedding:', error);
			return null;
		}
	}

	async searchSimilarChunks(query: string): Promise<DocumentChunk[]> {
		if (!this.genAI) return [];

		const queryEmbedding = await this.getEmbedding(query);
		if (!queryEmbedding) return [];

		const allChunks: DocumentChunk[] = [];

		for (const embeddingData of this.embeddingCache.values()) {
			for (const chunk of embeddingData.chunks) {
				if (chunk.embedding) {
					const similarity = this.cosineSimilarity(queryEmbedding, chunk.embedding);
					if (similarity >= this.settings.similarityThreshold) {
						allChunks.push({ ...chunk, similarity });
					}
				}
			}
		}

		return allChunks
			.sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
			.slice(0, this.settings.maxResults);
	}

	cosineSimilarity(a: number[], b: number[]): number {
		const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
		const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
		const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
		return dotProduct / (magnitudeA * magnitudeB);
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

class RAGSearchModal extends Modal {
	plugin: GeminiRAGPlugin;
	queryInput: HTMLInputElement;
	resultContainer: HTMLElement;
	isSearching: boolean = false;

	constructor(app: App, plugin: GeminiRAGPlugin) {
		super(app);
		this.plugin = plugin;
	}

	onOpen() {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.addClass('gemini-rag-modal');

		// Title
		const title = contentEl.createEl('h2', { text: 'Gemini RAG Search' });

		// Search input
		const inputContainer = contentEl.createDiv('search-input-container');
		this.queryInput = inputContainer.createEl('input', {
			type: 'text',
			placeholder: 'Ask a question about your vault...'
		});
		this.queryInput.addClass('search-input');

		const searchButton = inputContainer.createEl('button', { text: 'Search' });
		searchButton.addClass('search-button');

		// Result container
		this.resultContainer = contentEl.createDiv('result-container');

		// Event listeners
		searchButton.addEventListener('click', () => this.performSearch());
		this.queryInput.addEventListener('keypress', (e) => {
			if (e.key === 'Enter') {
				this.performSearch();
			}
		});

		// Focus on input
		setTimeout(() => this.queryInput.focus(), 100);
	}

	async performSearch() {
		const query = this.queryInput.value.trim();
		if (!query || this.isSearching) return;

		this.isSearching = true;
		this.resultContainer.empty();

		const loadingEl = this.resultContainer.createDiv('loading');
		loadingEl.setText('Searching and generating response...');

		try {
			// Show relevant chunks first
			const relevantChunks = await this.plugin.searchSimilarChunks(query);

			if (relevantChunks.length === 0) {
				this.showNoResults();
				return;
			}

			// Get AI response
			const response = await this.plugin.queryWithRAG(query);

			this.showResults(query, response, relevantChunks);
		} catch (error) {
			console.error('Search error:', error);
			this.showError('An error occurred while searching. Please check your API key and try again.');
		} finally {
			this.isSearching = false;
		}
	}

	showResults(query: string, response: string, chunks: DocumentChunk[]) {
		this.resultContainer.empty();

		// AI Response
		const responseSection = this.resultContainer.createDiv('response-section');
		responseSection.createEl('h3', { text: 'AI Response' });
		const responseEl = responseSection.createDiv('ai-response');
		this.renderMarkdown(response, responseEl);

		// Sources
		const sourcesSection = this.resultContainer.createDiv('sources-section');
		sourcesSection.createEl('h3', { text: 'Relevant Sources' });

		chunks.forEach((chunk, index) => {
			const sourceEl = sourcesSection.createDiv('source-item');

			const sourceHeader = sourceEl.createDiv('source-header');
			sourceHeader.createEl('strong', { text: chunk.fileName });
			sourceHeader.createEl('span', {
				text: ` (${(chunk.similarity! * 100).toFixed(1)}% match)`,
				cls: 'similarity-score'
			});

			const sourceContent = sourceEl.createDiv('source-content');
			sourceContent.setText(chunk.content.substring(0, 300) + (chunk.content.length > 300 ? '...' : ''));

			// Add click handler to open file
			sourceEl.addEventListener('click', () => {
				this.app.workspace.openLinkText(chunk.filePath, '');
				this.close();
			});
			sourceEl.addClass('clickable-source');
		});
	}

	showNoResults() {
		this.resultContainer.empty();
		const noResultsEl = this.resultContainer.createDiv('no-results');
		noResultsEl.setText('No relevant content found in your vault for this query.');
	}

	showError(message: string) {
		this.resultContainer.empty();
		const errorEl = this.resultContainer.createDiv('error-message');
		errorEl.setText(message);
	}

	async renderMarkdown(content: string, container: HTMLElement) {
		container.empty();
		await MarkdownRenderer.renderMarkdown(content, container, '', this.plugin);
	}

	formatResponse(response: string): string {
		// Simple formatting - convert newlines to <br> and preserve paragraphs
		return response
			.split('\n\n')
			.map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
			.join('');
	}

	onClose() {
		const { contentEl } = this;
		contentEl.empty();
	}
}

// ChatModal class removed - using sidebar only

class ChatView extends ItemView {
	plugin: GeminiRAGPlugin;
	chatHistory: ChatMessage[] = [];
	messageInput: HTMLInputElement;
	chatContainer: HTMLElement;
	inputContainer: HTMLElement;
	isProcessing: boolean = false;
	isResizing: boolean = false;
	resizeStartX: number = 0;
	resizeStartY: number = 0;
	resizeStartWidth: number = 0;
	resizeStartHeight: number = 0;
	mouseMoveHandler: (e: MouseEvent) => void;
	mouseUpHandler: () => void;

	constructor(app: App, plugin: GeminiRAGPlugin) {
		super(app);
		this.plugin = plugin;
		
		// Bind event handlers
		this.mouseMoveHandler = (e: MouseEvent) => this.handleResize(e);
		this.mouseUpHandler = () => this.stopResize();
	}

	onOpen() {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.addClass('gemini-chat-modal');

		// Add resize handles
		this.addResizeHandles(contentEl);

		// Title
		const header = contentEl.createDiv('chat-header');
		header.createEl('h2', { text: 'Gemini RAG Chat' });
		
		const clearButton = header.createEl('button', { text: 'Clear Chat' });
		clearButton.addClass('clear-chat-button');
		clearButton.addEventListener('click', () => this.clearChat());

		// Chat container
		this.chatContainer = contentEl.createDiv('chat-container');

		// Input container
		this.inputContainer = contentEl.createDiv('chat-input-container');
		
		this.messageInput = this.inputContainer.createEl('input', {
			type: 'text',
			placeholder: 'Ask anything about your vault...'
		});
		this.messageInput.addClass('chat-input');

		const sendButton = this.inputContainer.createEl('button', { text: 'Send' });
		sendButton.addClass('send-button');

		// Event listeners
		sendButton.addEventListener('click', () => this.sendMessage());
		this.messageInput.addEventListener('keypress', (e) => {
			if (e.key === 'Enter' && !e.shiftKey) {
				e.preventDefault();
				this.sendMessage();
			}
		});

		// Focus on input
		setTimeout(() => this.messageInput.focus(), 100);
		
		// Load chat history if exists
		this.loadChatHistory();
		this.renderChatHistory();
	}

	clearChat() {
		this.chatHistory = [];
		this.renderChatHistory();
		this.saveChatHistory();
	}

	async sendMessage() {
		const message = this.messageInput.value.trim();
		if (!message || this.isProcessing) return;

		this.isProcessing = true;
		this.messageInput.value = '';
		this.messageInput.disabled = true;

		// Add user message
		const userMessage: ChatMessage = {
			role: 'user',
			content: message,
			timestamp: Date.now()
		};
		this.chatHistory.push(userMessage);
		this.renderChatHistory();

		try {
			// Get relevant chunks
			const relevantChunks = await this.plugin.searchSimilarChunks(message);
			
			// Generate response with RAG
			const response = await this.plugin.queryWithRAG(message);
			
			// Add assistant message
			const assistantMessage: ChatMessage = {
				role: 'assistant',
				content: response,
				timestamp: Date.now(),
				sources: relevantChunks.length > 0 ? relevantChunks : undefined
			};
			this.chatHistory.push(assistantMessage);
			
		} catch (error) {
			console.error('Chat error:', error);
			const errorMessage: ChatMessage = {
				role: 'assistant',
				content: 'Sorry, I encountered an error processing your message. Please check your API key and try again.',
				timestamp: Date.now()
			};
			this.chatHistory.push(errorMessage);
		}

		this.renderChatHistory();
		this.saveChatHistory();
		this.isProcessing = false;
		this.messageInput.disabled = false;
		this.messageInput.focus();
	}

	renderChatHistory() {
		this.chatContainer.empty();

		if (this.chatHistory.length === 0) {
			const emptyState = this.chatContainer.createDiv('empty-chat');
			emptyState.createEl('p', { text: '👋 Start a conversation! Ask me anything about your vault.' });
			return;
		}

		this.chatHistory.forEach((message) => {
			const messageEl = this.chatContainer.createDiv(`message ${message.role}-message`);
			
			// Message header
			const messageHeader = messageEl.createDiv('message-header');
			messageHeader.createEl('span', { 
				text: message.role === 'user' ? 'You' : 'Gemini',
				cls: 'message-author'
			});
			messageHeader.createEl('span', { 
				text: new Date(message.timestamp).toLocaleTimeString(),
				cls: 'message-time'
			});

			// Message content
			const messageContent = messageEl.createDiv('message-content');
			if (message.role === 'assistant') {
				this.renderMarkdown(message.content, messageContent);
			} else {
				messageContent.setText(message.content);
			}

			// Sources for assistant messages
			if (message.role === 'assistant' && message.sources && message.sources.length > 0) {
				const sourcesEl = messageEl.createDiv('message-sources');
				sourcesEl.createEl('small', { text: 'Sources:' });
				
				const sourcesList = sourcesEl.createEl('ul');
				message.sources.forEach((source) => {
					const sourceItem = sourcesList.createEl('li');
					const sourceLink = sourceItem.createEl('a', {
						text: `${source.fileName} (${(source.similarity! * 100).toFixed(1)}%)`,
						href: '#'
					});
					sourceLink.addEventListener('click', (e) => {
						e.preventDefault();
						this.app.workspace.openLinkText(source.filePath, '');
					});
				});
			}
		});

		// Scroll to bottom
		this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
	}

	async renderMarkdown(content: string, container: HTMLElement) {
		container.empty();
		await MarkdownRenderer.renderMarkdown(content, container, '', this.plugin);
	}

	formatResponse(response: string): string {
		return response
			.split('\n\n')
			.map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
			.join('');
	}

	async loadChatHistory() {
		try {
			const data = await this.plugin.loadData();
			if (data && data.chatHistory) {
				this.chatHistory = data.chatHistory;
			}
		} catch (error) {
			console.error('Error loading chat history:', error);
		}
	}

	async saveChatHistory() {
		try {
			const existingData = await this.plugin.loadData() || {};
			existingData.chatHistory = this.chatHistory;
			await this.plugin.saveData(existingData);
		} catch (error) {
			console.error('Error saving chat history:', error);
		}
	}

	addResizeHandles(contentEl: HTMLElement) {
		// Right handle
		const rightHandle = contentEl.createDiv('resize-handle resize-handle-right');
		rightHandle.addEventListener('mousedown', (e) => this.startResize(e, 'right'));

		// Bottom handle
		const bottomHandle = contentEl.createDiv('resize-handle resize-handle-bottom');
		bottomHandle.addEventListener('mousedown', (e) => this.startResize(e, 'bottom'));

		// Bottom-right corner handle
		const cornerHandle = contentEl.createDiv('resize-handle resize-handle-corner');
		cornerHandle.addEventListener('mousedown', (e) => this.startResize(e, 'corner'));

		// Add global mouse events
		document.addEventListener('mousemove', this.mouseMoveHandler);
		document.addEventListener('mouseup', this.mouseUpHandler);
	}

	startResize(e: MouseEvent, direction: string) {
		e.preventDefault();
		e.stopPropagation();
		console.log('Starting resize in direction:', direction);
		
		this.isResizing = true;
		this.resizeStartX = e.clientX;
		this.resizeStartY = e.clientY;
		
		const modalEl = this.contentEl.closest('.modal') as HTMLElement;
		if (modalEl) {
			const rect = modalEl.getBoundingClientRect();
			this.resizeStartWidth = rect.width;
			this.resizeStartHeight = rect.height;
			modalEl.setAttribute('data-resize-direction', direction);
			console.log('Initial size:', this.resizeStartWidth, 'x', this.resizeStartHeight);
		}
		
		document.body.classList.add('is-resizing');
	}

	handleResize(e: MouseEvent) {
		if (!this.isResizing) return;

		const modalEl = this.contentEl.closest('.modal') as HTMLElement;
		if (!modalEl) return;

		const direction = modalEl.getAttribute('data-resize-direction');
		const deltaX = e.clientX - this.resizeStartX;
		const deltaY = e.clientY - this.resizeStartY;

		let newWidth = this.resizeStartWidth;
		let newHeight = this.resizeStartHeight;

		if (direction === 'right' || direction === 'corner') {
			newWidth = Math.max(400, this.resizeStartWidth + deltaX);
		}
		
		if (direction === 'bottom' || direction === 'corner') {
			newHeight = Math.max(300, this.resizeStartHeight + deltaY);
		}

		modalEl.style.width = `${newWidth}px`;
		modalEl.style.height = `${newHeight}px`;
		modalEl.style.maxWidth = 'none';
		modalEl.style.maxHeight = 'none';
	}

	stopResize() {
		if (!this.isResizing) return;
		
		this.isResizing = false;
		document.body.classList.remove('is-resizing');
		
		const modalEl = this.contentEl.closest('.modal') as HTMLElement;
		if (modalEl) {
			modalEl.removeAttribute('data-resize-direction');
		}
	}

	onClose() {
		const { contentEl } = this;
		contentEl.empty();
		this.saveChatHistory();
		
		// Clean up resize event listeners
		document.removeEventListener('mousemove', this.mouseMoveHandler);
		document.removeEventListener('mouseup', this.mouseUpHandler);
	}
}

class ChatView extends ItemView {
	plugin: GeminiRAGPlugin;
	chatHistory: ChatMessage[] = [];
	messageInput: HTMLInputElement;
	chatContainer: HTMLElement;
	inputContainer: HTMLElement;
	isProcessing: boolean = false;

	constructor(leaf: WorkspaceLeaf, plugin: GeminiRAGPlugin) {
		super(leaf);
		this.plugin = plugin;
	}

	getViewType() {
		return CHAT_VIEW_TYPE;
	}

	getDisplayText() {
		return "Gemini RAG Chat";
	}

	getIcon() {
		return "message-circle";
	}

	async onOpen() {
		const container = this.containerEl.children[1];
		container.empty();
		container.addClass('gemini-chat-view');

		// Header
		const header = container.createDiv('chat-view-header');
		header.createEl('h3', { text: 'Gemini RAG Chat' });
		
		const clearButton = header.createEl('button', { text: 'Clear' });
		clearButton.addClass('clear-chat-button');
		clearButton.addEventListener('click', () => this.clearChat());

		// Chat container
		this.chatContainer = container.createDiv('chat-view-container');

		// Input container
		this.inputContainer = container.createDiv('chat-view-input-container');
		
		this.messageInput = this.inputContainer.createEl('input', {
			type: 'text',
			placeholder: 'Ask anything about your vault...'
		});
		this.messageInput.addClass('chat-view-input');

		const sendButton = this.inputContainer.createEl('button', { text: 'Send' });
		sendButton.addClass('send-button');

		// Event listeners
		sendButton.addEventListener('click', () => this.sendMessage());
		this.messageInput.addEventListener('keypress', (e) => {
			if (e.key === 'Enter') {
				e.preventDefault();
				this.sendMessage();
			}
		});

		// Load and render chat history
		await this.loadChatHistory();
		this.renderChatHistory();
	}

	clearChat() {
		this.chatHistory = [];
		this.renderChatHistory();
		this.saveChatHistory();
	}

	async sendMessage() {
		const message = this.messageInput.value.trim();
		if (!message || this.isProcessing) return;

		this.isProcessing = true;
		this.messageInput.value = '';
		this.messageInput.disabled = true;

		// Add user message
		const userMessage: ChatMessage = {
			role: 'user',
			content: message,
			timestamp: Date.now()
		};
		this.chatHistory.push(userMessage);
		this.renderChatHistory();

		try {
			// Get relevant chunks and generate response
			const relevantChunks = await this.plugin.searchSimilarChunks(message);
			const response = await this.plugin.queryWithRAG(message);
			
			// Add assistant message
			const assistantMessage: ChatMessage = {
				role: 'assistant',
				content: response,
				timestamp: Date.now(),
				sources: relevantChunks.length > 0 ? relevantChunks : undefined
			};
			this.chatHistory.push(assistantMessage);
			
		} catch (error) {
			console.error('Chat error:', error);
			const errorMessage: ChatMessage = {
				role: 'assistant',
				content: 'Sorry, I encountered an error processing your message. Please check your API key and try again.',
				timestamp: Date.now()
			};
			this.chatHistory.push(errorMessage);
		}

		this.renderChatHistory();
		this.saveChatHistory();
		this.isProcessing = false;
		this.messageInput.disabled = false;
		this.messageInput.focus();
	}

	renderChatHistory() {
		this.chatContainer.empty();

		if (this.chatHistory.length === 0) {
			const emptyState = this.chatContainer.createDiv('empty-chat');
			emptyState.createEl('p', { text: '👋 Start a conversation! Ask me anything about your vault.' });
			return;
		}

		this.chatHistory.forEach((message) => {
			const messageEl = this.chatContainer.createDiv(`message ${message.role}-message`);
			
			// Message header
			const messageHeader = messageEl.createDiv('message-header');
			messageHeader.createEl('span', { 
				text: message.role === 'user' ? 'You' : 'Gemini',
				cls: 'message-author'
			});
			messageHeader.createEl('span', { 
				text: new Date(message.timestamp).toLocaleTimeString(),
				cls: 'message-time'
			});

			// Message content
			const messageContent = messageEl.createDiv('message-content');
			if (message.role === 'assistant') {
				this.renderMarkdown(message.content, messageContent);
			} else {
				messageContent.setText(message.content);
			}

			// Sources for assistant messages
			if (message.role === 'assistant' && message.sources && message.sources.length > 0) {
				const sourcesEl = messageEl.createDiv('message-sources');
				sourcesEl.createEl('small', { text: 'Sources:' });
				
				const sourcesList = sourcesEl.createEl('ul');
				message.sources.forEach((source) => {
					const sourceItem = sourcesList.createEl('li');
					const sourceLink = sourceItem.createEl('a', {
						text: `${source.fileName} (${(source.similarity! * 100).toFixed(1)}%)`,
						href: '#'
					});
					sourceLink.addEventListener('click', (e) => {
						e.preventDefault();
						this.app.workspace.openLinkText(source.filePath, '');
					});
				});
			}
		});

		// Scroll to bottom
		this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
	}

	async renderMarkdown(content: string, container: HTMLElement) {
		container.empty();
		await MarkdownRenderer.renderMarkdown(content, container, '', this.plugin);
	}

	formatResponse(response: string): string {
		return response
			.split('\n\n')
			.map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
			.join('');
	}

	async loadChatHistory() {
		try {
			const data = await this.plugin.loadData();
			if (data && data.chatHistory) {
				this.chatHistory = data.chatHistory;
			}
		} catch (error) {
			console.error('Error loading chat history:', error);
		}
	}

	async saveChatHistory() {
		try {
			const existingData = await this.plugin.loadData() || {};
			existingData.chatHistory = this.chatHistory;
			await this.plugin.saveData(existingData);
		} catch (error) {
			console.error('Error saving chat history:', error);
		}
	}

	async onClose() {
		await this.saveChatHistory();
	}
}

class GeminiRAGSettingTab extends PluginSettingTab {
	plugin: GeminiRAGPlugin;

	constructor(app: App, plugin: GeminiRAGPlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();

		containerEl.createEl('h2', { text: 'Gemini RAG Plugin Settings' });

		// API Key
		new Setting(containerEl)
			.setName('Gemini API Key')
			.setDesc('Your Google AI Studio API key. Get one from https://makersuite.google.com/app/apikey')
			.addText(text => text
				.setPlaceholder('Enter your API key')
				.setValue(this.plugin.settings.apiKey)
				.onChange(async (value) => {
					this.plugin.settings.apiKey = value;
					await this.plugin.saveSettings();
				}));

		// Embedding Model
		new Setting(containerEl)
			.setName('Embedding Model')
			.setDesc('The Gemini model to use for generating embeddings')
			.addDropdown(dropdown => dropdown
				.addOption('text-embedding-004', 'text-embedding-004 (Recommended)')
				.setValue(this.plugin.settings.embeddingModel)
				.onChange(async (value) => {
					this.plugin.settings.embeddingModel = value;
					await this.plugin.saveSettings();
				}));

		// Generative Model
		new Setting(containerEl)
			.setName('Generative Model')
			.setDesc('The Gemini model to use for generating responses')
			.addDropdown(dropdown => dropdown
				.addOption('gemini-1.5-flash', 'Gemini 1.5 Flash (Fast)')
				.addOption('gemini-1.5-pro', 'Gemini 1.5 Pro (Better quality)')
				.setValue(this.plugin.settings.generativeModel)
				.onChange(async (value) => {
					this.plugin.settings.generativeModel = value;
					await this.plugin.saveSettings();
				}));

		// Max Results
		new Setting(containerEl)
			.setName('Max Search Results')
			.setDesc('Maximum number of relevant chunks to include in context')
			.addSlider(slider => slider
				.setLimits(1, 20, 1)
				.setValue(this.plugin.settings.maxResults)
				.setDynamicTooltip()
				.onChange(async (value) => {
					this.plugin.settings.maxResults = value;
					await this.plugin.saveSettings();
				}));

		// Similarity Threshold
		new Setting(containerEl)
			.setName('Similarity Threshold')
			.setDesc('Minimum similarity score for including results (0.0 - 1.0)')
			.addSlider(slider => slider
				.setLimits(0.1, 1.0, 0.05)
				.setValue(this.plugin.settings.similarityThreshold)
				.setDynamicTooltip()
				.onChange(async (value) => {
					this.plugin.settings.similarityThreshold = value;
					await this.plugin.saveSettings();
				}));

		// Chunk Size
		new Setting(containerEl)
			.setName('Chunk Size')
			.setDesc('Size of text chunks for embedding (in characters)')
			.addSlider(slider => slider
				.setLimits(200, 2000, 100)
				.setValue(this.plugin.settings.chunkSize)
				.setDynamicTooltip()
				.onChange(async (value) => {
					this.plugin.settings.chunkSize = value;
					await this.plugin.saveSettings();
				}));

		// Chunk Overlap
		new Setting(containerEl)
			.setName('Chunk Overlap')
			.setDesc('Overlap between consecutive chunks (in characters)')
			.addSlider(slider => slider
				.setLimits(0, 500, 50)
				.setValue(this.plugin.settings.chunkOverlap)
				.setDynamicTooltip()
				.onChange(async (value) => {
					this.plugin.settings.chunkOverlap = value;
					await this.plugin.saveSettings();
				}));

		// Actions section
		containerEl.createEl('h3', { text: 'Actions' });

		// Rebuild embeddings button
		new Setting(containerEl)
			.setName('Rebuild Embeddings Database')
			.setDesc('Process all markdown files and create new embeddings. This may take several minutes.')
			.addButton(button => button
				.setButtonText('Rebuild Embeddings')
				.setCta()
				.onClick(async () => {
					await this.plugin.rebuildEmbeddings();
				}));

		// Stats
		const totalChunks = this.plugin.getTotalChunks();
		containerEl.createEl('p', {
			text: `Current database: ${totalChunks} chunks from ${this.plugin.embeddingCache.size} files`,
			cls: 'setting-item-description'
		});
	}
}


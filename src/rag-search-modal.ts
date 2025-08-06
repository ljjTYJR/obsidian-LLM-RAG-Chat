import { App, Modal, MarkdownRenderer } from 'obsidian';
import { DocumentChunk } from './types';
import type GeminiRAGPlugin from '../main';

export class RAGSearchModal extends Modal {
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
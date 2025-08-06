import { ItemView, WorkspaceLeaf, MarkdownRenderer } from 'obsidian';
import { CHAT_VIEW_TYPE, ChatMessage } from './types';
import type GeminiRAGPlugin from '../main';

export class ChatView extends ItemView {
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
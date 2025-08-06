import { App, PluginSettingTab, Setting } from 'obsidian';
import type GeminiRAGPlugin from './main';

export class GeminiRAGSettingTab extends PluginSettingTab {
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
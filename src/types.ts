export const CHAT_VIEW_TYPE = "gemini-rag-chat-view";

export interface GeminiRAGSettings {
	apiKey: string;
	embeddingModel: string;
	generativeModel: string;
	maxResults: number;
	similarityThreshold: number;
	chunkSize: number;
	chunkOverlap: number;
}

export const DEFAULT_SETTINGS: GeminiRAGSettings = {
	apiKey: '',
	embeddingModel: 'text-embedding-004',
	generativeModel: 'gemini-1.5-flash-latest',
	maxResults: 5,
	similarityThreshold: 0.7,
	chunkSize: 1000,
	chunkOverlap: 200
}

export interface DocumentChunk {
	content: string;
	filePath: string;
	fileName: string;
	embedding?: number[];
	similarity?: number;
}

export interface ChatMessage {
	role: 'user' | 'assistant';
	content: string;
	timestamp: number;
	sources?: DocumentChunk[];
}

export interface EmbeddingData {
	chunks: DocumentChunk[];
	lastUpdated: number;
}
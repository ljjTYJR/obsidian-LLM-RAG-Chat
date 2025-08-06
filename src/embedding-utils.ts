import { TFile } from 'obsidian';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { DocumentChunk, EmbeddingData, GeminiRAGSettings } from './types';

export class EmbeddingUtils {
	static chunkText(text: string, chunkSize: number, overlap: number): string[] {
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

	static async getEmbedding(text: string, genAI: GoogleGenerativeAI, settings: GeminiRAGSettings): Promise<number[] | null> {
		if (!genAI) return null;

		try {
			const model = genAI.getGenerativeModel({ model: settings.embeddingModel });
			const result = await model.embedContent(text);
			return result.embedding.values;
		} catch (error) {
			console.error('Error getting embedding:', error);
			return null;
		}
	}

	static cosineSimilarity(a: number[], b: number[]): number {
		const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
		const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
		const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
		return dotProduct / (magnitudeA * magnitudeB);
	}

	static async processFileForEmbeddings(
		file: TFile,
		vault: any,
		genAI: GoogleGenerativeAI,
		settings: GeminiRAGSettings
	): Promise<EmbeddingData | null> {
		try {
			const content = await vault.read(file);
			const chunks = this.chunkText(content, settings.chunkSize, settings.chunkOverlap);

			const documentChunks: DocumentChunk[] = [];

			for (const chunk of chunks) {
				if (chunk.trim().length < 50) continue; // Skip very short chunks

				const embedding = await this.getEmbedding(chunk, genAI, settings);
				if (embedding) {
					documentChunks.push({
						content: chunk,
						filePath: file.path,
						fileName: file.name,
						embedding: embedding
					});
				}
			}

			return {
				chunks: documentChunks,
				lastUpdated: Date.now()
			};
		} catch (error) {
			console.error(`Error processing file ${file.path}:`, error);
			return null;
		}
	}

	static searchSimilarChunks(
		query: string,
		queryEmbedding: number[],
		embeddingCache: Map<string, EmbeddingData>,
		settings: GeminiRAGSettings
	): DocumentChunk[] {
		if (!queryEmbedding) return [];

		const allChunks: DocumentChunk[] = [];

		for (const embeddingData of embeddingCache.values()) {
			for (const chunk of embeddingData.chunks) {
				if (chunk.embedding) {
					const similarity = this.cosineSimilarity(queryEmbedding, chunk.embedding);
					if (similarity >= settings.similarityThreshold) {
						allChunks.push({ ...chunk, similarity });
					}
				}
			}
		}

		return allChunks
			.sort((a, b) => (b.similarity || 0) - (a.similarity || 0))
			.slice(0, settings.maxResults);
	}
}
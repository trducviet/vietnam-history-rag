import { config as dotenvConfig } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
dotenvConfig({ path: resolve(__dirname, '../../.env') });

/** Application configuration derived from environment and defaults */
export const config = {
  /** OpenAI API key */
  openaiApiKey: process.env.OPENAI_API_KEY || '',

  /** Path to the dataset pack */
  dataPath: resolve(
    __dirname,
    '../..',
    process.env.DATA_PATH || '../vietnam_history_dataset_runtime_optimal_pack'
  ),

  /** @deprecated Legacy combined cache path — use event/synthesis paths instead */
  embeddingCachePath: resolve(
    __dirname,
    '../..',
    process.env.EMBEDDING_CACHE_PATH || './cache/embeddings.json'
  ),

  /** Path to cache event embeddings */
  eventEmbeddingCachePath: resolve(
    __dirname,
    '../..',
    './cache/embeddings.event.json'
  ),

  /** Path to cache synthesis embeddings */
  synthesisEmbeddingCachePath: resolve(
    __dirname,
    '../..',
    './cache/embeddings.synthesis.json'
  ),

  /** Server port */
  port: parseInt(process.env.PORT || '3000', 10),

  /** Embedding model */
  embeddingModel: 'text-embedding-3-large' as const,

  /** Embedding dimensions */
  embeddingDimensions: 3072,

  /** Router/planner model (cost-effective) */
  routerModel: 'gpt-5-mini' as const,

  /** Final answer generation model (high quality) */
  generationModel: 'gpt-5.4' as const,

  /** Default top-k for retrieval */
  defaultTopK: 10,

  /** Vector weight in hybrid fusion (α for vector, 1-α for BM25). Override with HYBRID_VECTOR_WEIGHT env. */
  vectorWeight: parseFloat(process.env.HYBRID_VECTOR_WEIGHT || '0.6'),
} as const;

/** Validate that critical config is present */
export function validateConfig(): void {
  if (!config.openaiApiKey) {
    console.warn(
      '⚠️  OPENAI_API_KEY not set. Embedding generation and LLM calls will fail.\n' +
      '   Copy .env.example to .env and set your key.'
    );
  }
}
